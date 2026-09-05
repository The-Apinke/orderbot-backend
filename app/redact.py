import hmac
import hashlib
import os
import re

_SECRET = os.getenv("PII_HASH_SECRET", "").encode()

# Same shape as the phone detector in routes.py:validate_phones_in_message,
# reused deliberately so "what counts as a phone number" is defined once.
_PHONE_PATTERN = re.compile(r'\+?\d{7,13}')


def _normalize(digits: str) -> str:
    """Collapse +234 international format to local 0xxx format first, so
    the same real number hashes identically regardless of how the customer
    typed it — same normalization validate_phones_in_message already uses."""
    if len(digits) == 13 and digits.startswith("234"):
        return "0" + digits[3:]
    return digits


def _hash_digits(digits: str) -> str:
    """HMAC, not a bare hash — an 11-digit Nigerian mobile number is too
    small a search space for a bare hash to resist brute-forcing."""
    normalized = _normalize(digits)
    return hmac.new(_SECRET, normalized.encode(), hashlib.sha256).hexdigest()[:12]


def redact_phone_numbers(text: str) -> str:
    """Replace phone-shaped substrings with a stable, non-reversible token.

    Only touches the digits — everything else in the message is left exactly
    as written, so tone and phrasing stay inspectable for quality review.
    The same input number always produces the same token, so repeat
    customers are still correlatable across traces without ever storing
    the real number in Langfuse.
    """
    if not text:
        return text

    def _sub(match):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 7:
            return raw
        return f"[PHONE:{_hash_digits(digits)}]"

    return _PHONE_PATTERN.sub(_sub, text)
