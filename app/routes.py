from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, field_validator
from app.database import supabase
from app.agent import get_streaming_response, get_welcome_message, has_packaging_instructions, extract_order_inventory, check_inventory_vs_response
from fastapi.responses import StreamingResponse
import json
import os
import io
from openai import OpenAI

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: list = []

@router.get("/menu")
def get_menu():
    response = supabase.table("menu_items").select("*").eq("available", True).execute()
    
    menu = {}
    for item in response.data:
        category = item["category"]
        if category not in menu:
            menu[category] = []
        menu[category].append({
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "price": item["price"]
        })
    
    return {"menu": menu}

@router.get("/chat/welcome")
def welcome():
    message = get_welcome_message()
    return {"message": message}

@router.post("/chat/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    contents = await audio.read()
    audio_file = io.BytesIO(contents)
    audio_file.name = audio.filename or "audio.webm"
    transcript = get_openai_client().audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return {"transcript": transcript.text}

@router.post("/chat")
async def chat(request: ChatRequest):
    menu_response = supabase.table("menu_items").select("*").eq("available", True).execute()
    
    menu = {}
    for item in menu_response.data:
        category = item["category"]
        if category not in menu:
            menu[category] = []
        menu[category].append({
            "name": item["name"],
            "description": item["description"],
            "price": item["price"]
        })

    messages = request.conversation_history + [
        {"role": "user", "content": request.message}
    ]

    async def generate():
        full_reply = ""

        # Extract inventory before streaming if complex packaging order detected
        inventory = {}
        if has_packaging_instructions(request.message):
            inventory = extract_order_inventory(request.message)

        with get_streaming_response(messages, menu) as stream:
            for text in stream.text_stream:
                full_reply += text
                yield f"data: {json.dumps({'token': text})}\n\n"

        # Python code verification — no AI involved
        if inventory:
            unassigned = check_inventory_vs_response(inventory, full_reply)
            if unassigned:
                items_str = ", ".join(f"{qty}x {item}" for item, qty in unassigned.items())
                verb = "it doesn't appear" if len(unassigned) == 1 else "they don't appear"
                pronoun = "that" if len(unassigned) == 1 else "those"
                correction = (
                    f"\n\nHold on — you ordered {items_str} but {verb} "
                    f"in any package above. Which package should {pronoun} go into?"
                )
                yield f"data: {json.dumps({'token': correction})}\n\n"
                full_reply += correction

        updated_history = messages + [{"role": "assistant", "content": full_reply}]
        yield f"data: {json.dumps({'done': True, 'conversation_history': updated_history})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

class OrderRequest(BaseModel):
    customer_name: str
    customer_phone: str
    items: list
    total_price: float
    notes: str = ""

    @field_validator('customer_name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Customer name cannot be empty')
        return v.strip()

    @field_validator('customer_phone')
    @classmethod
    def phone_must_be_valid(cls, v):
        digits = ''.join(filter(str.isdigit, v))
        if len(digits) < 11:
            raise ValueError('Phone number must have at least 11 digits')
        return v.strip()

    @field_validator('total_price')
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Total price must be greater than zero')
        return v

    @field_validator('items')
    @classmethod
    def items_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Order must have at least one item')
        return v

@router.post("/orders")
def create_order(order: OrderRequest):
    response = supabase.table("orders").insert({
        "customer_name": order.customer_name,
        "customer_phone": order.customer_phone,
        "items": order.items,
        "total_price": order.total_price,
        "status": "pending",
        "notes": order.notes
    }).execute()

    return {
        "message": "Order placed successfully!",
        "order_id": response.data[0]["id"]
    }

@router.get("/orders")
def get_orders():
    response = supabase.table("orders").select("*").order("created_at", desc=True).execute()
    return {"orders": response.data}

class OrderStatusUpdate(BaseModel):
    status: str

@router.patch("/orders/{order_id}")
def update_order_status(order_id: str, update: OrderStatusUpdate):
    response = supabase.table("orders").update({
        "status": update.status
    }).eq("id", order_id).execute()
    
    return {"message": "Order status updated"}