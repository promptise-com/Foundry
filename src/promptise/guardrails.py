"""Promptise Security Scanner — local ML-powered input/output guardrails.

Provides :class:`PromptiseSecurityScanner`, a unified scanner that detects
prompt injection attacks, PII leakage, toxic content, and credential
exposure using local transformer models and comprehensive regex patterns.
No external API calls — all detection runs locally.

Example::

    from promptise.guardrails import PromptiseSecurityScanner

    scanner = PromptiseSecurityScanner()

    # Use with build_agent
    agent = await build_agent(..., guardrails=scanner)

    # Or standalone
    report = await scanner.scan_text("my input text")
    print(report.passed)
    print(report.findings)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

logger = logging.getLogger("promptise.guardrails")

__all__ = [
    # Scanner
    "PromptiseSecurityScanner",
    # Detectors (composable API)
    "InjectionDetector",
    "PIIDetector",
    "CredentialDetector",
    "ContentSafetyDetector",
    "NERDetector",
    "CustomRule",
    # Categories (typed enums)
    "PIICategory",
    "CredentialCategory",
    # Results
    "SecurityFinding",
    "ScanReport",
    "Severity",
    "Action",
    "GuardrailViolation",
    # Agent integration
    "wrap_tools_with_guardrails",
]


# ═══════════════════════════════════════════════════════════════════════
# Typed category enums — use these with enable_pii / enable_credentials
# for IDE autocomplete and type-safe configuration
# ═══════════════════════════════════════════════════════════════════════


class PIICategory(str, Enum):
    """PII detection categories.  Pass a set of these to
    ``PromptiseSecurityScanner(enable_pii={...})`` to enable only
    specific PII types.

    Example::

        scanner = PromptiseSecurityScanner(
            enable_pii={PIICategory.CREDIT_CARDS, PIICategory.SSN, PIICategory.EMAIL},
        )
    """

    # Payment cards
    CREDIT_CARDS = "credit_cards"
    CVV = "cvv"
    CARD_EXPIRY = "card_expiry"

    # US government IDs
    SSN = "ssn"
    US_PASSPORT = "us_passport"
    ITIN = "itin"
    EIN = "ein"

    # International government IDs
    UK_IDS = "uk_ids"
    CANADA_SIN = "ca_sin"
    FRANCE_INSEE = "fr_insee"
    ITALY_CF = "it_codice_fiscale"
    SPAIN_DNI = "es_dni"
    GERMANY_ID = "de_id"
    NETHERLANDS_BSN = "nl_bsn"
    AUSTRALIA_IDS = "au_ids"
    BRAZIL_IDS = "br_ids"
    INDIA_IDS = "in_ids"
    SINGAPORE_NRIC = "sg_nric"
    SOUTH_KOREA_RRN = "kr_rrn"
    JAPAN_MY_NUMBER = "jp_my_number"
    MEXICO_CURP = "mx_curp"
    SOUTH_AFRICA_ID = "za_id"

    # Driver's licenses
    DRIVERS_LICENSE = "drivers_license"

    # Contact
    EMAIL = "email"
    PHONE = "phone"
    POSTAL_CODE = "postal_code"

    # Financial
    IBAN = "iban"
    SWIFT = "swift"
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    CRYPTO_WALLET = "crypto_wallet"

    # Healthcare
    NPI = "npi"
    DEA = "dea"
    MEDICAL_RECORD = "medical_record"
    DIAGNOSIS_CODE = "diagnosis_code"
    DRUG_CODE = "drug_code"
    BLOOD_TYPE = "blood_type"

    # Biographic
    DATE_OF_BIRTH = "date_of_birth"

    # Network / infra
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"

    # Credentials in text
    PASSWORD = "password"
    SECRET = "secret"

    # Vehicle
    VIN = "vin"
    LICENSE_PLATE = "license_plate"

    # Convenience groups — use ALL for everything
    ALL = "all"


class CredentialCategory(str, Enum):
    """Credential detection categories.  Pass a set of these to
    ``PromptiseSecurityScanner(enable_credentials={...})``.

    Example::

        scanner = PromptiseSecurityScanner(
            enable_credentials={
                CredentialCategory.AWS,
                CredentialCategory.OPENAI,
                CredentialCategory.GITHUB,
            },
        )
    """

    # AI / ML
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"

    # Cloud
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ALIBABA = "alibaba"
    DIGITALOCEAN = "digitalocean"
    FLY_IO = "fly_io"

    # Git platforms
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"

    # Payments
    STRIPE = "stripe"
    SQUARE = "square"
    PLAID = "plaid"
    BRAINTREE = "braintree"
    TWILIO = "twilio"
    FLUTTERWAVE = "flutterwave"

    # Communication
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"

    # Email services
    SENDGRID = "sendgrid"
    MAILGUN = "mailgun"
    MAILCHIMP = "mailchimp"
    SENDINBLUE = "sendinblue"

    # Monitoring
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    GRAFANA = "grafana"
    SENTRY = "sentry"
    DYNATRACE = "dynatrace"

    # Auth / secrets
    HASHICORP = "hashicorp"
    DOPPLER = "doppler"
    PULUMI = "pulumi"
    ONEPASSWORD = "onepassword"

    # Commerce
    SHOPIFY = "shopify"

    # Collaboration
    ATLASSIAN = "atlassian"
    NOTION = "notion"
    POSTMAN = "postman"
    AIRTABLE = "airtable"
    TYPEFORM = "typeform"

    # Package registries
    NPM = "npm"
    PYPI = "pypi"
    RUBYGEMS = "rubygems"

    # Infrastructure
    TERRAFORM = "terraform"
    DATABRICKS = "databricks"
    PLANETSCALE = "planetscale"
    BUILDKITE = "buildkite"
    HEROKU = "heroku"
    FIREBASE = "firebase"

    # Tokens / keys
    JWT = "jwt"
    BEARER = "bearer"
    PRIVATE_KEY = "private_key"
    AGE_KEY = "age_key"

    # Database URLs
    DATABASE_URL = "database_url"

    # Other
    MAPBOX = "mapbox"
    DUFFEL = "duffel"
    EASYPOST = "easypost"
    SHIPPO = "shippo"
    FRAMEIO = "frameio"
    CLOUDINARY = "cloudinary"
    PASSWORD_URL = "password_url"

    ALL = "all"


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════


class Severity(str, Enum):
    """Severity level of a security finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    """Action to take when a finding is detected."""

    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


@dataclass
class SecurityFinding:
    """A single detection result from a scanner.

    Attributes:
        detector: Which detection head found this (injection/pii/toxicity/credential).
        category: Specific sub-category (e.g. ``"credit_card_visa"``, ``"aws_access_key"``).
        severity: How severe this finding is.
        confidence: Model confidence or 1.0 for regex matches.
        matched_text: The text span that matched.
        start: Character offset in original text.
        end: Character offset in original text.
        action: What should happen (block/redact/warn).
        description: Human-readable explanation.
        metadata: Extra information (model scores, pattern name, etc.).
    """

    detector: str
    category: str
    severity: Severity
    confidence: float
    matched_text: str
    start: int
    end: int
    action: Action
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanReport:
    """Complete scan result with all findings and metadata.

    Attributes:
        passed: True if no findings have action=BLOCK.
        findings: All detections from all scanners.
        duration_ms: Total scan time in milliseconds.
        scanners_run: Which detection heads ran.
        text_length: Length of scanned text.
        redacted_text: Text with PII/credentials replaced (output scans).
        user_id: Owning user (from :class:`~promptise.agent.CallerContext`)
            for audit attribution.  ``None`` when the scan ran outside of
            an invocation context.
        session_id: Conversation session attached to this scan, when
            available (from ``caller.metadata['session_id']``).
        caller_roles: The caller's roles at scan time, captured so that
            downstream audit log entries can rationalize the decision
            even after the caller context has moved on.
        scanners_skipped: Detection heads that were enabled but could not
            run, mapped to the reason (``transformers`` missing, model
            failed to load, Ollama unreachable, ...).  A skipped head is
            never listed in ``scanners_run``.  With ``fail_open=False``
            (the default) each skipped head also adds a ``BLOCK`` finding,
            so the scan fails.
    """

    passed: bool
    findings: list[SecurityFinding]
    duration_ms: float
    scanners_run: list[str]
    text_length: int
    redacted_text: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    caller_roles: tuple[str, ...] = field(default_factory=tuple)
    scanners_skipped: dict[str, str] = field(default_factory=dict)

    @property
    def blocked(self) -> list[SecurityFinding]:
        """Findings that caused a block."""
        return [f for f in self.findings if f.action == Action.BLOCK]

    @property
    def redacted(self) -> list[SecurityFinding]:
        """Findings that were redacted."""
        return [f for f in self.findings if f.action == Action.REDACT]

    @property
    def warnings(self) -> list[SecurityFinding]:
        """Findings that are warnings only."""
        return [f for f in self.findings if f.action == Action.WARN]


class GuardrailViolation(Exception):
    """Raised when input or output is blocked by guardrails.

    Attributes:
        report: The full scan report.
        direction: ``"input"``, ``"output"`` or ``"tool"`` (a tool result).
    """

    def __init__(self, report: ScanReport, direction: str = "input") -> None:
        self.report = report
        self.direction = direction
        blocked = report.blocked
        details = "; ".join(f.description for f in blocked[:3])
        super().__init__(
            f"Guardrail violation ({direction}): {len(blocked)} blocked finding(s). {details}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Luhn algorithm for credit card validation
# ═══════════════════════════════════════════════════════════════════════


def _luhn_check(number: str) -> bool:
    """Validate a number string using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ═══════════════════════════════════════════════════════════════════════
# PII regex patterns (25+ patterns)
# ═══════════════════════════════════════════════════════════════════════


# Each pattern: (name, category, compiled_re, severity, description, group)
_PII_PATTERNS: list[tuple[str, str, re.Pattern[str], Severity, str, str]] = []


def _pii(
    name: str, category: str, pattern: str, severity: Severity, desc: str, *, group: str = ""
) -> None:
    _PII_PATTERNS.append((name, category, re.compile(pattern), severity, desc, group))


# ── Credit / Debit cards (validated with Luhn) ──
_pii(
    "visa",
    "credit_card_visa",
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",
    Severity.CRITICAL,
    "Visa credit card number",
    group="credit_cards",
)
_pii(
    "mastercard",
    "credit_card_mastercard",
    r"\b5[1-5][0-9]{14}\b",
    Severity.CRITICAL,
    "Mastercard credit card number",
    group="credit_cards",
)
_pii(
    "mastercard_2series",
    "credit_card_mastercard",
    r"\b2(?:2[2-9][1-9]|2[3-9]\d|[3-6]\d{2}|7[01]\d|720)\d{12}\b",
    Severity.CRITICAL,
    "Mastercard 2-series credit card number",
    group="credit_cards",
)
_pii(
    "amex",
    "credit_card_amex",
    r"\b3[47][0-9]{13}\b",
    Severity.CRITICAL,
    "American Express credit card number",
    group="credit_cards",
)
_pii(
    "discover",
    "credit_card_discover",
    r"\b6(?:011|5[0-9]{2})[0-9]{12}\b",
    Severity.CRITICAL,
    "Discover credit card number",
    group="credit_cards",
)
_pii(
    "diners_club",
    "credit_card_diners",
    r"\b3(?:0[0-5]|[68][0-9])[0-9]{11}\b",
    Severity.CRITICAL,
    "Diners Club credit card number",
    group="credit_cards",
)
_pii(
    "jcb",
    "credit_card_jcb",
    r"\b(?:2131|1800|35\d{3})\d{11}\b",
    Severity.CRITICAL,
    "JCB credit card number",
    group="credit_cards",
)
_pii(
    "unionpay",
    "credit_card_unionpay",
    r"\b62[0-9]{14,17}\b",
    Severity.CRITICAL,
    "UnionPay credit card number",
    group="credit_cards",
)
_pii(
    "maestro",
    "credit_card_maestro",
    r"\b(?:5018|5020|5038|5893|6304|6759|6761|6762|6763)\d{8,15}\b",
    Severity.CRITICAL,
    "Maestro debit card number",
    group="credit_cards",
)
_pii(
    "cc_formatted",
    "credit_card",
    r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b",
    Severity.CRITICAL,
    "Formatted credit card number",
    group="credit_cards",
)
_pii(
    "cc_cvv",
    "cvv",
    r"(?i)\b(?:cvv|cvc|cvv2|cvc2|cid)\s*[:=]?\s*\d{3,4}\b",
    Severity.CRITICAL,
    "Card verification value (CVV/CVC)",
    group="cvv",
)
_pii(
    "cc_expiry",
    "card_expiry",
    r"(?i)(?:exp(?:ir(?:y|ation))?|valid\s*(?:thru|through|until))\s*[:=]?\s*(?:0[1-9]|1[0-2])\s*[/\-]\s*(?:\d{2}|\d{4})",
    Severity.HIGH,
    "Card expiration date",
    group="card_expiry",
)

# ── US Government IDs ──
_pii(
    "ssn",
    "ssn",
    r"\b(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}\b",
    Severity.CRITICAL,
    "US Social Security Number",
    group="ssn",
)
_pii(
    "ssn_no_dash",
    "ssn",
    r"\b(?!666|000|9\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}\b",
    Severity.HIGH,
    "US SSN without dashes",
    group="ssn",
)
_pii(
    "us_passport",
    "passport",
    r"(?i)(?:passport)\s*#?\s*[:=]?\s*([A-Z]{1,2}[0-9]{6,9})",
    Severity.HIGH,
    "US passport number (contextual)",
    group="us_passport",
)
_pii(
    "us_itin",
    "itin",
    r"\b9\d{2}-[7-9]\d-\d{4}\b",
    Severity.CRITICAL,
    "US Individual Taxpayer Identification Number (ITIN)",
    group="itin",
)
_pii(
    "us_ein",
    "ein",
    r"(?i)(?:ein|employer\s*id(?:entification)?)\s*#?\s*[:=]?\s*(\d{2}-\d{7})",
    Severity.HIGH,
    "US Employer ID (contextual)",
    group="ein",
)

# ── International Government IDs ──
_pii(
    "uk_nino",
    "national_insurance",
    r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b",
    Severity.CRITICAL,
    "UK National Insurance Number (NINO)",
    group="uk_ids",
)
_pii(
    "uk_passport",
    "passport",
    r"(?i)(?:passport)\s*#?\s*[:=]?\s*(\d{9})\b",
    Severity.HIGH,
    "UK passport number (contextual)",
    group="uk_ids",
)
_pii(
    "uk_drivers",
    "drivers_license",
    r"\b[A-Z]{5}\d{6}[A-Z]{2}\d{5}\b",
    Severity.HIGH,
    "UK driver's license number",
    group="uk_ids",
)
_pii(
    "uk_nhs",
    "nhs_number",
    r"(?i)(?:nhs)\s*#?\s*[:=]?\s*(\d{3}\s?\d{3}\s?\d{4})",
    Severity.HIGH,
    "UK NHS number (contextual)",
    group="uk_ids",
)
_pii(
    "ca_sin",
    "social_insurance",
    r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b",
    Severity.CRITICAL,
    "Canadian Social Insurance Number (SIN)",
    group="ca_sin",
)
_pii(
    "de_personalausweis",
    "national_id",
    r"\b[CFGHJKLMNPRTVWXYZ0-9]{9}\b",
    Severity.MEDIUM,
    "German ID card (Personalausweis) number",
    group="de_id",
)
_pii(
    "fr_insee",
    "national_id",
    r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
    Severity.CRITICAL,
    "French INSEE/Social Security number",
    group="fr_insee",
)
_pii(
    "it_codice_fiscale",
    "national_id",
    r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b",
    Severity.HIGH,
    "Italian Codice Fiscale (tax ID)",
    group="it_codice_fiscale",
)
_pii(
    "es_dni", "national_id", r"\b\d{8}[A-Z]\b", Severity.HIGH, "Spanish DNI number", group="es_dni"
)
_pii(
    "nl_bsn",
    "national_id",
    r"(?i)(?:bsn|burgerservicenummer)\s*#?\s*[:=]?\s*(\d{9})",
    Severity.MEDIUM,
    "Dutch BSN (contextual)",
    group="nl_bsn",
)
_pii(
    "au_tfn",
    "tax_file_number",
    r"\b\d{3}\s?\d{3}\s?\d{3}\b",
    Severity.HIGH,
    "Australian Tax File Number (TFN)",
    group="au_ids",
)
_pii(
    "au_medicare",
    "medicare",
    r"(?i)(?:medicare)\s*#?\s*[:=]?\s*([2-6]\d{3}\s?\d{5}\s?\d)",
    Severity.HIGH,
    "Australian Medicare number (contextual)",
    group="au_ids",
)
_pii(
    "br_cpf",
    "cpf",
    r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    Severity.CRITICAL,
    "Brazilian CPF (tax ID)",
    group="br_ids",
)
_pii(
    "br_cnpj",
    "cnpj",
    r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
    Severity.HIGH,
    "Brazilian CNPJ (company tax ID)",
    group="br_ids",
)
_pii(
    "in_aadhaar",
    "aadhaar",
    r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
    Severity.CRITICAL,
    "Indian Aadhaar number",
    group="in_ids",
)
_pii(
    "in_pan",
    "pan_card",
    r"(?i)(?:pan|permanent\s*account)\s*#?\s*[:=]?\s*([A-Z]{5}\d{4}[A-Z])",
    Severity.HIGH,
    "Indian PAN card (contextual)",
    group="in_ids",
)
_pii(
    "sg_nric",
    "national_id",
    r"\b[STFGM]\d{7}[A-Z]\b",
    Severity.HIGH,
    "Singapore NRIC/FIN number",
    group="sg_nric",
)
_pii(
    "kr_rrn",
    "resident_registration",
    r"\b\d{6}-[1-4]\d{6}\b",
    Severity.CRITICAL,
    "South Korean Resident Registration Number",
    group="kr_rrn",
)
_pii(
    "jp_my_number",
    "my_number",
    r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    Severity.CRITICAL,
    "Japanese My Number (Individual Number)",
    group="jp_my_number",
)
_pii(
    "mx_curp",
    "curp",
    r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b",
    Severity.HIGH,
    "Mexican CURP (population registry key)",
    group="mx_curp",
)
_pii(
    "za_id",
    "national_id",
    r"\b\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4}[01]\d{2}\b",
    Severity.HIGH,
    "South African ID number",
    group="za_id",
)

# ── Driver's licenses (US states — common formats) ──
_pii(
    "dl_california",
    "drivers_license",
    r"\b[A-Z]\d{7}\b",
    Severity.HIGH,
    "California driver's license (A + 7 digits)",
    group="drivers_license",
)
_pii(
    "dl_new_york",
    "drivers_license",
    r"(?i)(?:driver'?s?\s*(?:license|licence|lic)|DL)\s*#?\s*[:=]?\s*(\d{3}\s?\d{3}\s?\d{3})",
    Severity.MEDIUM,
    "New York driver's license (contextual)",
    group="drivers_license",
)
_pii(
    "dl_florida",
    "drivers_license",
    r"\b[A-Z]\d{3}-\d{3}-\d{2}-\d{3}-\d\b",
    Severity.HIGH,
    "Florida driver's license",
    group="drivers_license",
)
_pii(
    "dl_texas",
    "drivers_license",
    r"(?i)(?:driver'?s?\s*(?:license|licence|lic)|DL)\s*#?\s*[:=]?\s*(\d{8})\b",
    Severity.LOW,
    "Texas driver's license (contextual)",
    group="drivers_license",
)

# ── Contact information ──
_pii(
    "email",
    "email",
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    Severity.MEDIUM,
    "Email address",
    group="email",
)
# Global phone — requires + country code prefix (high precision for international)
_pii(
    "phone_global",
    "phone",
    r"(?<!\w)\+[1-9]\d{0,2}[-.\s]?\(?\d{1,5}\)?(?:[-.\s]?\d{1,5}){1,4}(?!\w)",
    Severity.MEDIUM,
    "Phone number (international)",
    group="phone",
)
# US specific (high precision — area code + 7 digits)
# US phone — requires at least one separator (dash, dot, space, parens) to avoid matching bare 10-digit numbers
_pii(
    "phone_us",
    "phone",
    r"(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}\b",
    Severity.MEDIUM,
    "US phone number",
    group="phone",
)

# ── Addresses & location ──
_pii(
    "us_zip",
    "zip_code",
    r"(?i)(?:zip|postal)\s*(?:code)?\s*[:=]?\s*(\d{5}(?:-\d{4})?)\b",
    Severity.LOW,
    "US ZIP code (contextual)",
    group="postal_code",
)
_pii(
    "uk_postcode",
    "postcode",
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
    Severity.LOW,
    "UK postcode",
    group="postal_code",
)
_pii(
    "ca_postal",
    "postal_code",
    r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b",
    Severity.LOW,
    "Canadian postal code",
    group="postal_code",
)

# ── Financial ──
_pii(
    "iban",
    "iban",
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,18})?\b",
    Severity.HIGH,
    "International Bank Account Number (IBAN)",
    group="iban",
)
_pii(
    "swift",
    "swift_bic",
    r"(?i)(?:swift|bic)\s*(?:code)?\s*[:=]\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)",
    Severity.MEDIUM,
    "SWIFT/BIC code (contextual)",
    group="swift",
)
_pii(
    "routing",
    "routing_number",
    r"\b0[0-9]\d{7}\b",
    Severity.MEDIUM,
    "US bank routing number (starts with 0)",
    group="routing_number",
)
_pii(
    "us_bank_account",
    "bank_account",
    r"(?i)(?:account|acct)\s*#?\s*[:=]?\s*\d{8,17}",
    Severity.HIGH,
    "US bank account number (contextual)",
    group="bank_account",
)
_pii(
    "bitcoin_address",
    "crypto_wallet",
    r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b",
    Severity.MEDIUM,
    "Bitcoin wallet address",
    group="crypto_wallet",
)
_pii(
    "ethereum_address",
    "crypto_wallet",
    r"\b0x[a-fA-F0-9]{40}\b",
    Severity.MEDIUM,
    "Ethereum wallet address",
    group="crypto_wallet",
)

# ── Healthcare / Medical ──
_pii(
    "npi",
    "npi",
    r"(?i)(?:npi|national\s*provider)\s*#?\s*[:=]?\s*\d{10}",
    Severity.HIGH,
    "National Provider Identifier (NPI)",
    group="npi",
)
_pii(
    "dea",
    "dea_number",
    r"(?i)(?:dea)\s*#?\s*[:=]?\s*([ABCDEFGHJKLMNPRSTUX][A-Z9]\d{7})",
    Severity.HIGH,
    "DEA registration number (contextual)",
    group="dea",
)
_pii(
    "mrn",
    "medical_record",
    r"(?i)(?:mrn|medical\s*record)\s*#?\s*[:=]?\s*[A-Z0-9\-]{6,20}",
    Severity.HIGH,
    "Medical record number (contextual)",
    group="medical_record",
)
_pii(
    "icd10",
    "diagnosis_code",
    r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b",
    Severity.MEDIUM,
    "ICD-10 diagnosis code",
    group="diagnosis_code",
)
_pii(
    "ndc",
    "drug_code",
    r"\b\d{4,5}-\d{3,4}-\d{1,2}\b",
    Severity.MEDIUM,
    "National Drug Code (NDC)",
    group="drug_code",
)
# Contextual: a bare "A-" or "O+" is far more often an ID prefix ("order
# A-1001") or a grade than a blood type, so the keyword is required.
_pii(
    "blood_type",
    "medical",
    r"(?i)\bblood[\s_-]*(?:type|group)\s*(?:is\s+)?[:=,]?\s*(?:AB|A|B|O)\s?"
    r"(?:[+-](?![\w+-])|(?:Rh\s*)?(?:pos(?:itive)?|neg(?:ative)?)\b)",
    Severity.LOW,
    "Blood type (contextual)",
    group="blood_type",
)

# ── Date of birth (contextual) ──
_pii(
    "dob_us",
    "date_of_birth",
    r"(?i)(?:dob|date\s*of\s*birth|birth\s*date|born)\s*[:=]?\s*(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}",
    Severity.HIGH,
    "Date of birth (US format MM/DD/YYYY)",
    group="date_of_birth",
)
_pii(
    "dob_eu",
    "date_of_birth",
    r"(?i)(?:dob|date\s*of\s*birth|birth\s*date|born)\s*[:=]?\s*(?:0[1-9]|[12]\d|3[01])[/\-.](?:0[1-9]|1[0-2])[/\-.](?:19|20)\d{2}",
    Severity.HIGH,
    "Date of birth (EU format DD/MM/YYYY)",
    group="date_of_birth",
)

# ── Network / Infrastructure ──
_pii(
    "ipv4",
    "ip_address",
    r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b",
    Severity.LOW,
    "IPv4 address",
    group="ip_address",
)
_pii(
    "ipv6",
    "ip_address",
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    Severity.LOW,
    "IPv6 address (full form)",
    group="ip_address",
)
_pii(
    "mac_address",
    "mac_address",
    r"\b(?:[0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b",
    Severity.LOW,
    "MAC address",
    group="mac_address",
)

# ── Usernames / Passwords (contextual) ──
_pii(
    "password_field",
    "password",
    r"(?i)(?:password|passwd|pwd)\s*[:=]\s*\S{4,}",
    Severity.CRITICAL,
    "Password in plaintext (contextual)",
    group="password",
)
_pii(
    "secret_field",
    "secret",
    r"(?i)(?:secret|private[_\-]?key|api[_\-]?secret)\s*[:=]\s*\S{8,}",
    Severity.CRITICAL,
    "Secret/private key in plaintext (contextual)",
    group="secret",
)

# ── Vehicle ──
_pii(
    "vin",
    "vehicle_id",
    r"(?i)(?:vin|vehicle\s*id)\s*#?\s*[:=]?\s*([A-HJ-NPR-Z0-9]{17})",
    Severity.MEDIUM,
    "Vehicle Identification Number (contextual)",
    group="vin",
)
_pii(
    "us_plate",
    "license_plate",
    r"(?i)(?:plate|tag|registration)\s*#?\s*[:=]\s*([A-Z0-9]{2,4}[-\s][A-Z0-9]{2,4}[-\s]?[A-Z0-9]{0,4})",
    Severity.LOW,
    "License plate (contextual)",
    group="license_plate",
)


# ═══════════════════════════════════════════════════════════════════════
# Credential regex patterns (50+ patterns from gitleaks/trufflehog)
# ═══════════════════════════════════════════════════════════════════════


# Each pattern: (name, category, compiled_re, severity, description, group)
_CRED_PATTERNS: list[tuple[str, str, re.Pattern[str], Severity, str, str]] = []


def _cred(
    name: str, category: str, pattern: str, severity: Severity, desc: str, *, group: str = ""
) -> None:
    _CRED_PATTERNS.append((name, category, re.compile(pattern), severity, desc, group))


# ── AWS ──
_cred(
    "aws_access_key",
    "aws_access_key",
    r"(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}",
    Severity.CRITICAL,
    "AWS access key ID",
    group="aws",
)
_cred(
    "aws_secret_key",
    "aws_secret_key",
    r"(?i)(?:aws_secret_access_key|aws_secret)\s*[:=]\s*[A-Za-z0-9/+=]{40}",
    Severity.CRITICAL,
    "AWS secret access key",
    group="aws",
)
_cred(
    "aws_mws",
    "aws_mws_token",
    r"amzn\.mws\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    Severity.CRITICAL,
    "Amazon MWS auth token",
    group="aws",
)

# ── Google/GCP ──
_cred(
    "gcp_api_key",
    "google_api_key",
    r"AIza[0-9A-Za-z\-_]{35}",
    Severity.HIGH,
    "Google API key",
    group="gcp",
)
_cred(
    "gcp_oauth",
    "google_oauth",
    r"[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com",
    Severity.HIGH,
    "Google OAuth client ID",
    group="gcp",
)
_cred(
    "gcp_sa",
    "gcp_service_account",
    r'"type"\s*:\s*"service_account"',
    Severity.CRITICAL,
    "GCP service account JSON",
    group="gcp",
)
_cred(
    "firebase",
    "firebase_url",
    r"[a-z0-9.-]+\.firebaseio\.com",
    Severity.MEDIUM,
    "Firebase Realtime Database URL",
    group="firebase",
)

# ── Azure ──
_cred(
    "azure_key",
    "azure_key",
    r"(?i)(?:AccountKey|azure[_-]?(?:storage|account)[_-]?key)\s*[:=]\s*[A-Za-z0-9+/=]{44,88}",
    Severity.CRITICAL,
    "Azure storage account key",
    group="azure",
)

# ── GitHub ──
_cred(
    "github_pat",
    "github_pat",
    r"ghp_[a-zA-Z0-9]{36}",
    Severity.CRITICAL,
    "GitHub personal access token (classic)",
    group="github",
)
_cred(
    "github_pat_fine",
    "github_pat_fine",
    r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}",
    Severity.CRITICAL,
    "GitHub fine-grained personal access token",
    group="github",
)
_cred(
    "github_oauth",
    "github_oauth",
    r"gho_[a-zA-Z0-9]{36}",
    Severity.HIGH,
    "GitHub OAuth access token",
    group="github",
)
_cred(
    "github_app",
    "github_app_token",
    r"ghu_[a-zA-Z0-9]{36}",
    Severity.HIGH,
    "GitHub App user token",
    group="github",
)
_cred(
    "github_refresh",
    "github_refresh_token",
    r"ghr_[a-zA-Z0-9]{36}",
    Severity.HIGH,
    "GitHub refresh token",
    group="github",
)

# ── GitLab ──
_cred(
    "gitlab_pat",
    "gitlab_pat",
    r"glpat-[a-zA-Z0-9_\-]{20}",
    Severity.CRITICAL,
    "GitLab personal access token",
    group="gitlab",
)
_cred(
    "gitlab_ci",
    "gitlab_ci_token",
    r"glci-[0-9a-zA-Z_\-]{20}",
    Severity.HIGH,
    "GitLab CI token",
    group="gitlab",
)
_cred(
    "gitlab_deploy",
    "gitlab_deploy_token",
    r"gldt-[0-9a-zA-Z_\-]{20}",
    Severity.HIGH,
    "GitLab deploy token",
    group="gitlab",
)

# ── Payment processors ──
_cred(
    "stripe_live",
    "stripe_secret_key",
    r"sk_live_[0-9a-zA-Z]{24,}",
    Severity.CRITICAL,
    "Stripe live secret key",
    group="stripe",
)
_cred(
    "stripe_test",
    "stripe_test_key",
    r"sk_test_[0-9a-zA-Z]{24,}",
    Severity.HIGH,
    "Stripe test secret key",
    group="stripe",
)
_cred(
    "stripe_restricted",
    "stripe_restricted_key",
    r"rk_live_[0-9a-zA-Z]{24,}",
    Severity.CRITICAL,
    "Stripe restricted key",
    group="stripe",
)
_cred(
    "square",
    "square_access_token",
    r"sq0atp-[0-9A-Za-z\-_]{22}",
    Severity.CRITICAL,
    "Square access token",
    group="square",
)
_cred(
    "twilio",
    "twilio_api_key",
    r"SK[0-9a-fA-F]{32}",
    Severity.HIGH,
    "Twilio API key",
    group="twilio",
)

# ── Communication platforms ──
_cred(
    "slack_bot",
    "slack_bot_token",
    r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[0-9a-zA-Z]{24}",
    Severity.CRITICAL,
    "Slack bot token",
    group="slack",
)
_cred(
    "slack_user",
    "slack_user_token",
    r"xoxp-[0-9]{10,13}-[0-9]{10,13}-[0-9]{10,13}-[0-9a-zA-Z]{32}",
    Severity.CRITICAL,
    "Slack user token",
    group="slack",
)
_cred(
    "slack_webhook",
    "slack_webhook",
    r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]{8,}/B[a-zA-Z0-9_]{8,}/[a-zA-Z0-9_]{24}",
    Severity.HIGH,
    "Slack webhook URL",
    group="slack",
)
_cred(
    "discord_token",
    "discord_bot_token",
    r"[MN][A-Za-z0-9_\-]{23,25}\.[A-Za-z0-9_\-]{6,7}\.[A-Za-z0-9_\-]{27,}",
    Severity.CRITICAL,
    "Discord bot token",
    group="discord",
)
_cred(
    "discord_webhook",
    "discord_webhook",
    r"https://discord(?:app)?\.com/api/webhooks/[0-9]{17,19}/[A-Za-z0-9_\-]{60,68}",
    Severity.HIGH,
    "Discord webhook URL",
    group="discord",
)
_cred(
    "telegram",
    "telegram_bot_token",
    r"[0-9]{8,10}:[A-Za-z0-9_\-]{35}",
    Severity.CRITICAL,
    "Telegram bot token",
    group="telegram",
)

# ── Email services ──
_cred(
    "sendgrid",
    "sendgrid_api_key",
    r"SG\.[a-zA-Z0-9_\-]{22}\.[a-zA-Z0-9_\-]{43}",
    Severity.CRITICAL,
    "SendGrid API key",
    group="sendgrid",
)
_cred(
    "mailgun",
    "mailgun_api_key",
    r"key-[0-9a-zA-Z]{32}",
    Severity.HIGH,
    "Mailgun API key",
    group="mailgun",
)
_cred(
    "mailchimp",
    "mailchimp_api_key",
    r"[0-9a-f]{32}-us[0-9]{1,2}",
    Severity.HIGH,
    "MailChimp API key",
    group="mailchimp",
)

# ── Monitoring / observability ──
_cred(
    "datadog",
    "datadog_api_key",
    r"(?i)dd[_-]?api[_-]?key\s*[:=]\s*\w{32}",
    Severity.HIGH,
    "Datadog API key",
    group="datadog",
)
_cred(
    "newrelic",
    "newrelic_key",
    r"NRAK-[A-Z0-9]{27}",
    Severity.HIGH,
    "New Relic API key",
    group="newrelic",
)

# ── Auth tokens ──
_cred(
    "jwt",
    "jwt_token",
    r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
    Severity.HIGH,
    "JWT token",
    group="jwt",
)
_cred(
    "bearer",
    "bearer_token",
    r"(?i)(?:bearer|authorization)\s*[:=]\s*[A-Za-z0-9\-._~+/]+=*",
    Severity.HIGH,
    "Bearer/Authorization token",
    group="bearer",
)

# ── Cryptographic keys ──
_cred(
    "private_key",
    "private_key",
    r"-----BEGIN\s(?:RSA|DSA|EC|OPENSSH|PGP)\sPRIVATE\sKEY-----",
    Severity.CRITICAL,
    "Private key (RSA/DSA/EC/SSH/PGP)",
    group="private_key",
)
_cred(
    "age_secret",
    "age_secret_key",
    r"AGE-SECRET-KEY-1[QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L]{58}",
    Severity.CRITICAL,
    "Age encryption secret key",
    group="age_key",
)

# ── Database connection strings ──
_cred(
    "pg_conn",
    "postgres_connection",
    r"(?i)postgres(?:ql)?://[^\s]{10,}",
    Severity.CRITICAL,
    "PostgreSQL connection string",
    group="database_url",
)
_cred(
    "mysql_conn",
    "mysql_connection",
    r"(?i)mysql://[^\s]{10,}",
    Severity.CRITICAL,
    "MySQL connection string",
    group="database_url",
)
_cred(
    "mongo_conn",
    "mongodb_connection",
    r"mongodb(?:\+srv)?://[^\s]{10,}",
    Severity.CRITICAL,
    "MongoDB connection string",
    group="database_url",
)
_cred(
    "redis_conn",
    "redis_connection",
    r"redis://[^\s]{10,}",
    Severity.HIGH,
    "Redis connection string",
    group="database_url",
)

# ── Infrastructure ──
_cred(
    "digitalocean",
    "digitalocean_token",
    r"dop_v1_[a-f0-9]{64}",
    Severity.CRITICAL,
    "DigitalOcean personal access token",
    group="digitalocean",
)
_cred(
    "heroku",
    "heroku_api_key",
    r"(?i)heroku[_-]?api[_-]?key\s*[:=]\s*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    Severity.HIGH,
    "Heroku API key",
    group="heroku",
)
_cred(
    "npm_token", "npm_token", r"npm_[A-Za-z0-9]{36}", Severity.HIGH, "npm access token", group="npm"
)
_cred(
    "pypi_token",
    "pypi_api_token",
    r"pypi-[A-Za-z0-9_\-]{100,}",
    Severity.HIGH,
    "PyPI API token",
    group="pypi",
)

# ── Passwords in URLs ──
_cred(
    "password_url",
    "password_in_url",
    r"(?i)(?:https?|ftp)://[a-zA-Z0-9._%+\-]+:[a-zA-Z0-9._%+\-]+@",
    Severity.CRITICAL,
    "Password embedded in URL",
    group="password_url",
)
_cred(
    "cloudinary",
    "cloudinary_url",
    r"cloudinary://[a-zA-Z0-9:/@._\-]+",
    Severity.HIGH,
    "Cloudinary URL with credentials",
    group="cloudinary",
)

# ── AI / ML services (from gitleaks 41k★) ──
_cred(
    "openai_key",
    "openai_api_key",
    r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{20,}T3BlbkFJ[A-Za-z0-9_\-]{20,}",
    Severity.CRITICAL,
    "OpenAI API key",
    group="openai",
)
_cred(
    "openai_legacy",
    "openai_api_key",
    r"sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}",
    Severity.CRITICAL,
    "OpenAI API key (legacy format)",
    group="openai",
)
_cred(
    "anthropic_key",
    "anthropic_api_key",
    r"sk-ant-api03-[a-zA-Z0-9_\-]{93}AA",
    Severity.CRITICAL,
    "Anthropic API key",
    group="anthropic",
)
_cred(
    "anthropic_admin",
    "anthropic_admin_key",
    r"sk-ant-admin01-[a-zA-Z0-9_\-]{93}AA",
    Severity.CRITICAL,
    "Anthropic admin API key",
    group="anthropic",
)
_cred(
    "huggingface",
    "huggingface_token",
    r"hf_[a-zA-Z]{34}",
    Severity.HIGH,
    "HuggingFace access token",
    group="huggingface",
)
_cred(
    "huggingface_org",
    "huggingface_org_token",
    r"api_org_[a-zA-Z]{34}",
    Severity.HIGH,
    "HuggingFace organization API token",
    group="huggingface",
)
_cred(
    "replicate",
    "replicate_api_token",
    r"r8_[A-Za-z0-9]{36}",
    Severity.HIGH,
    "Replicate API token",
    group="replicate",
)

# ── E-Commerce / Shopify (from gitleaks) ──
_cred(
    "shopify_access",
    "shopify_access_token",
    r"shpat_[a-fA-F0-9]{32}",
    Severity.CRITICAL,
    "Shopify access token",
    group="shopify",
)
_cred(
    "shopify_custom",
    "shopify_custom_app_token",
    r"shpca_[a-fA-F0-9]{32}",
    Severity.HIGH,
    "Shopify custom app access token",
    group="shopify",
)
_cred(
    "shopify_private",
    "shopify_private_app_token",
    r"shppa_[a-fA-F0-9]{32}",
    Severity.HIGH,
    "Shopify private app access token",
    group="shopify",
)
_cred(
    "shopify_secret",
    "shopify_shared_secret",
    r"shpss_[a-fA-F0-9]{32}",
    Severity.HIGH,
    "Shopify shared secret",
    group="shopify",
)

# ── Cloud providers (from gitleaks) ──
_cred(
    "alibaba_key",
    "alibaba_access_key",
    r"LTAI[a-zA-Z0-9]{20}",
    Severity.CRITICAL,
    "Alibaba Cloud access key ID",
    group="alibaba",
)
_cred(
    "azure_ad",
    "azure_ad_client_secret",
    r"[a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.\-]{31,34}",
    Severity.CRITICAL,
    "Azure AD client secret",
    group="azure",
)
_cred(
    "do_oauth",
    "digitalocean_oauth_token",
    r"doo_v1_[a-f0-9]{64}",
    Severity.CRITICAL,
    "DigitalOcean OAuth token",
    group="digitalocean",
)
_cred(
    "fly_io",
    "fly_access_token",
    r"fo1_[\w\-]{43}",
    Severity.HIGH,
    "Fly.io access token",
    group="fly_io",
)

# ── Auth / Identity (from gitleaks) ──
_cred(
    "vault_service",
    "hashicorp_vault_token",
    r"hvs\.[\w\-]{90,120}",
    Severity.CRITICAL,
    "HashiCorp Vault service token",
    group="hashicorp",
)
_cred(
    "vault_batch",
    "hashicorp_vault_batch_token",
    r"hvb\.[\w\-]{138,300}",
    Severity.CRITICAL,
    "HashiCorp Vault batch token",
    group="hashicorp",
)
_cred(
    "terraform",
    "terraform_api_token",
    r"[a-z0-9]{14}\.atlasv1\.[a-z0-9\-_=]{60,70}",
    Severity.CRITICAL,
    "Terraform Cloud API token",
    group="hashicorp",
)
_cred(
    "doppler",
    "doppler_api_token",
    r"dp\.pt\.[a-zA-Z0-9]{43}",
    Severity.HIGH,
    "Doppler API token",
    group="doppler",
)
_cred(
    "pulumi",
    "pulumi_api_token",
    r"pul-[a-f0-9]{40}",
    Severity.HIGH,
    "Pulumi API token",
    group="pulumi",
)
_cred(
    "onepassword_sa",
    "onepassword_service_account",
    r"ops_eyJ[a-zA-Z0-9+/]{250,}={0,3}",
    Severity.CRITICAL,
    "1Password service account token",
    group="onepassword",
)

# ── Atlassian / Jira (from gitleaks) ──
_cred(
    "atlassian_v2",
    "atlassian_api_token",
    r"ATATT3[A-Za-z0-9_\-=]{186}",
    Severity.CRITICAL,
    "Atlassian API token v2",
    group="atlassian",
)

# ── Monitoring (from gitleaks) ──
_cred(
    "grafana_api",
    "grafana_api_key",
    r"eyJrIjoi[A-Za-z0-9]{70,400}={0,3}",
    Severity.HIGH,
    "Grafana API key",
    group="grafana",
)
_cred(
    "grafana_cloud",
    "grafana_cloud_token",
    r"glc_[A-Za-z0-9+/]{32,400}={0,3}",
    Severity.HIGH,
    "Grafana Cloud API token",
    group="grafana",
)
_cred(
    "grafana_sa",
    "grafana_service_account_token",
    r"glsa_[A-Za-z0-9]{32}_[A-Fa-f0-9]{8}",
    Severity.HIGH,
    "Grafana service account token",
    group="grafana",
)
_cred(
    "sentry_user",
    "sentry_user_token",
    r"sntryu_[a-f0-9]{64}",
    Severity.HIGH,
    "Sentry user token",
    group="sentry",
)
_cred(
    "newrelic_user",
    "newrelic_user_api_key",
    r"NRAK-[a-z0-9]{27}",
    Severity.HIGH,
    "New Relic user API key",
    group="newrelic",
)
_cred(
    "newrelic_browser",
    "newrelic_browser_api_token",
    r"NRJS-[a-f0-9]{19}",
    Severity.MEDIUM,
    "New Relic browser API token",
    group="newrelic",
)
_cred(
    "dynatrace",
    "dynatrace_api_token",
    r"dt0c01\.[a-zA-Z0-9]{24}\.[a-zA-Z0-9]{64}",
    Severity.HIGH,
    "Dynatrace API token",
    group="dynatrace",
)

# ── Payments / Finance (from gitleaks/trufflehog) ──
_cred(
    "square_oauth",
    "square_oauth_secret",
    r"sq0csp-[0-9A-Za-z\-_]{43}",
    Severity.CRITICAL,
    "Square OAuth secret",
    group="square",
)
_cred(
    "plaid",
    "plaid_api_token",
    r"access-(?:sandbox|development|production)-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    Severity.CRITICAL,
    "Plaid API token",
    group="plaid",
)
_cred(
    "braintree",
    "braintree_access_token",
    r"access_token\$production\$[0-9a-z]{16}\$[0-9a-f]{32}",
    Severity.CRITICAL,
    "Braintree access token",
    group="braintree",
)
_cred(
    "flutterwave_secret",
    "flutterwave_secret_key",
    r"FLWSECK_TEST-[a-h0-9]{32}-X",
    Severity.HIGH,
    "Flutterwave secret key",
    group="flutterwave",
)

# ── Collaboration (from gitleaks) ──
_cred(
    "notion",
    "notion_integration_token",
    r"ntn_[0-9]{11}[A-Za-z0-9]{32}[A-Za-z0-9]{3}",
    Severity.HIGH,
    "Notion integration token",
    group="notion",
)
_cred(
    "postman",
    "postman_api_token",
    r"PMAK-[a-fA-F0-9]{24}-[a-fA-F0-9]{34}",
    Severity.HIGH,
    "Postman API token",
    group="postman",
)
_cred(
    "airtable_pat",
    "airtable_personal_access_token",
    r"pat[a-zA-Z0-9]{14}\.[a-f0-9]{64}",
    Severity.HIGH,
    "Airtable personal access token",
    group="airtable",
)
_cred(
    "typeform",
    "typeform_api_token",
    r"tfp_[a-z0-9\-_.=]{59}",
    Severity.MEDIUM,
    "Typeform API token",
    group="typeform",
)

# ── Package registries (from gitleaks) ──
_cred(
    "rubygems",
    "rubygems_api_token",
    r"rubygems_[a-f0-9]{48}",
    Severity.HIGH,
    "RubyGems API token",
    group="rubygems",
)
_cred(
    "databricks",
    "databricks_api_token",
    r"dapi[a-f0-9]{32}",
    Severity.HIGH,
    "Databricks API token",
    group="databricks",
)
_cred(
    "planetscale_token",
    "planetscale_api_token",
    r"pscale_tkn_[\w=.\-]{32,64}",
    Severity.HIGH,
    "PlanetScale API token",
    group="planetscale",
)
_cred(
    "planetscale_pw",
    "planetscale_password",
    r"pscale_pw_[\w=.\-]{32,64}",
    Severity.HIGH,
    "PlanetScale password",
    group="planetscale",
)
_cred(
    "buildkite",
    "buildkite_agent_token",
    r"bkua_[a-f0-9]{40}",
    Severity.HIGH,
    "Buildkite agent token",
    group="buildkite",
)
_cred(
    "sendinblue",
    "sendinblue_api_token",
    r"xkeysib-[a-f0-9]{64}-[a-zA-Z0-9]{16}",
    Severity.HIGH,
    "Sendinblue/Brevo API token",
    group="sendinblue",
)

# ── Misc (from gitleaks) ──
_cred(
    "mapbox",
    "mapbox_api_token",
    r"pk\.[a-z0-9]{60}\.[a-z0-9]{22}",
    Severity.MEDIUM,
    "Mapbox API token",
    group="mapbox",
)
_cred(
    "duffel",
    "duffel_api_token",
    r"duffel_(?:test|live)_[a-zA-Z0-9_\-=]{43}",
    Severity.HIGH,
    "Duffel API token",
    group="duffel",
)
_cred(
    "easypost",
    "easypost_api_key",
    r"EZAK[a-zA-Z0-9]{54}",
    Severity.HIGH,
    "EasyPost API key",
    group="easypost",
)
_cred(
    "shippo",
    "shippo_api_token",
    r"shippo_(?:live|test)_[a-fA-F0-9]{40}",
    Severity.HIGH,
    "Shippo API token",
    group="shippo",
)
_cred(
    "frameio",
    "frameio_api_token",
    r"fio-u-[a-zA-Z0-9\-_=]{64}",
    Severity.HIGH,
    "Frame.io API token",
    group="frameio",
)
_cred(
    "google_oauth_access",
    "google_oauth_access_token",
    r"ya29\.[0-9A-Za-z\-_]+",
    Severity.HIGH,
    "Google OAuth access token",
    group="gcp",
)
_cred(
    "aws_appsync",
    "aws_appsync_key",
    r"da2-[a-z0-9]{26}",
    Severity.HIGH,
    "AWS AppSync GraphQL key",
    group="aws",
)


# ═══════════════════════════════════════════════════════════════════════
# Model loader (lazy, cached)
# ═══════════════════════════════════════════════════════════════════════

_model_cache: dict[str, Any] = {}


def _load_classifier(model_name: str) -> Any:
    """Load a HuggingFace text-classification pipeline (cached)."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        import warnings as _w

        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*resume_download.*")
            _w.filterwarnings("ignore", message=".*UNEXPECTED.*")
            from transformers import pipeline as hf_pipeline

            pipe = hf_pipeline(
                "text-classification",
                model=model_name,
                truncation=True,
                max_length=512,
            )
        _model_cache[model_name] = pipe
        logger.info("Loaded security model: %s", model_name)
        return pipe
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for ML-based guardrails. "
            "Install with: pip install transformers torch\n"
            f"Missing: {exc}"
        ) from exc


# ═══════════════════════════════════════════════════════════════════════
# PromptiseSecurityScanner
# ═══════════════════════════════════════════════════════════════════════


_DEFAULT_INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
_DEFAULT_TOXICITY_MODEL = "unitary/toxic-bert"

# Classifier window, in characters.  The models read at most 512 tokens, so
# long text is scanned as overlapping windows rather than cut off.  512
# characters stays well inside the token limit for ordinary text, and the
# 256-character overlap means any span up to 256 characters long (a whole
# injected instruction, say) lands intact in at least one window.
_CLASSIFIER_WINDOW = 512
_CLASSIFIER_STRIDE = 256
# GLiNER truncates its input at about 384 words, and Llama Guard / Azure
# take a few thousand characters per request.
_NER_WINDOW, _NER_STRIDE = 1500, 1200
_SAFETY_LOCAL_WINDOW, _SAFETY_LOCAL_STRIDE = 4000, 3500
_SAFETY_AZURE_WINDOW, _SAFETY_AZURE_STRIDE = 10000, 9000


def _windows(text: str, size: int, stride: int) -> list[tuple[int, str]]:
    """Split *text* into ``(offset, chunk)`` windows that cover all of it.

    Consecutive windows overlap by ``size - stride`` characters.  Text no
    longer than *size* is a single window.
    """
    if len(text) <= size:
        return [(0, text)]
    windows: list[tuple[int, str]] = []
    start = 0
    while True:
        windows.append((start, text[start : start + size]))
        if start + size >= len(text):
            return windows
        start += stride


def _top_prediction(item: Any) -> dict[str, Any]:
    """Return the top ``{label, score}`` dict from one pipeline result.

    A pipeline called on a list returns one dict per input, or one list of
    dicts per input when ``top_k`` is set.
    """
    if isinstance(item, list):
        return item[0] if item else {}
    return item if isinstance(item, dict) else {}


# ═══════════════════════════════════════════════════════════════════════
# Detector classes — composable detection heads
# ═══════════════════════════════════════════════════════════════════════


class InjectionDetector:
    """Detect prompt injection attacks using a local DeBERTa model.

    Args:
        model: HuggingFace model ID or local directory path.
        threshold: Confidence threshold for blocking (0.0-1.0).

    Example::

        InjectionDetector()
        InjectionDetector(model="/models/local/deberta", threshold=0.9)
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_INJECTION_MODEL,
        threshold: float = 0.85,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def warmup(self) -> None:
        """Pre-load the model."""
        _load_classifier(self.model)


class PIIDetector:
    """Detect PII using regex patterns with Luhn validation for credit cards.

    69 built-in patterns covering credit cards (12 issuers), government IDs
    (22+ countries), contact info, financial data, healthcare, and more.

    Args:
        categories: Set of :class:`PIICategory` to enable.  Defaults to all.
        action: What to do on output (default: REDACT).
        exclude: Pattern names to skip.

    Example::

        PIIDetector()  # all PII
        PIIDetector(categories={PIICategory.CREDIT_CARDS, PIICategory.SSN})
        PIIDetector(exclude={"blood_type", "ip_address"})
    """

    def __init__(
        self,
        *,
        categories: set[PIICategory] | None = None,
        action: Action = Action.REDACT,
        exclude: set[str] | None = None,
    ) -> None:
        if categories and PIICategory.ALL in categories:
            self.groups: set[str] | None = None
        elif categories:
            self.groups = {c.value for c in categories}
        else:
            self.groups = None  # all
        self.action = action
        self.exclude = exclude or set()


class CredentialDetector:
    """Detect leaked credentials using 96 regex patterns from gitleaks/trufflehog.

    Covers API keys for 60+ services, database URLs, private keys, and tokens.

    Args:
        categories: Set of :class:`CredentialCategory` to enable.  Defaults to all.
        action: What to do on output (default: REDACT).
        exclude: Pattern names to skip.

    Example::

        CredentialDetector()  # all credentials
        CredentialDetector(categories={CredentialCategory.AWS, CredentialCategory.OPENAI})
    """

    def __init__(
        self,
        *,
        categories: set[CredentialCategory] | None = None,
        action: Action = Action.REDACT,
        exclude: set[str] | None = None,
    ) -> None:
        if categories and CredentialCategory.ALL in categories:
            self.groups: set[str] | None = None
        elif categories:
            self.groups = {c.value for c in categories}
        else:
            self.groups = None  # all
        self.action = action
        self.exclude = exclude or set()


class ContentSafetyDetector:
    """Detect harmful content across 13 safety categories.

    Uses Meta's Llama Guard (local via Ollama) or Azure AI Content Safety (cloud).

    **Local** (default): Requires Ollama running with ``llama-guard3`` model.
    **Azure**: Requires ``AZURE_CONTENT_SAFETY_KEY`` and endpoint URL.

    Categories: violent crimes, non-violent crimes, sex crimes, child exploitation,
    defamation, specialized advice, privacy, intellectual property, weapons,
    hate speech, self-harm, sexual content, elections.

    Args:
        provider: ``"local"`` for Ollama/Llama Guard, ``"azure"`` for Azure AI.
        azure_endpoint: Azure Content Safety endpoint URL.
        azure_key: Azure API key (supports ``${ENV_VAR}`` syntax).
        threshold: Severity threshold (0.0-1.0).
        action: What to do on detection (default: BLOCK).

    Example::

        ContentSafetyDetector()  # local via Ollama
        ContentSafetyDetector(provider="azure", azure_endpoint="https://...")
    """

    def __init__(
        self,
        *,
        provider: str = "local",
        azure_endpoint: str | None = None,
        azure_key: str | None = None,
        threshold: float = 0.5,
        action: Action = Action.BLOCK,
    ) -> None:
        self.provider = provider
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key
        self.threshold = threshold
        self.action = action
        self._model: Any = None

    # Llama Guard 13 safety categories
    CATEGORIES: dict[str, str] = {
        "S1": "Violent crimes",
        "S2": "Non-violent crimes",
        "S3": "Sex-related crimes",
        "S4": "Child sexual exploitation",
        "S5": "Defamation",
        "S6": "Specialized advice",
        "S7": "Privacy",
        "S8": "Intellectual property",
        "S9": "Indiscriminate weapons",
        "S10": "Hate",
        "S11": "Suicide and self-harm",
        "S12": "Sexual content",
        "S13": "Elections",
    }

    _OLLAMA_URL = "http://localhost:11434"
    _OLLAMA_MODEL = "llama-guard3"

    async def scan(self, text: str) -> list[dict[str, Any]]:
        """Scan text for content safety violations.

        Long text is checked in overlapping windows, so content past the
        provider's per-request limit is scanned too.

        Returns:
            List of dicts with ``category``, ``label``, ``confidence``.

        Raises:
            Exception: When the provider cannot be reached or answers with
                an error.  The scanner turns this into a blocked scan
                (or a skipped one with ``fail_open=True``); it never
                reads as "safe".
        """
        if self.provider == "azure":
            size, stride, scan_one = _SAFETY_AZURE_WINDOW, _SAFETY_AZURE_STRIDE, self._scan_azure
        else:
            size, stride, scan_one = _SAFETY_LOCAL_WINDOW, _SAFETY_LOCAL_STRIDE, self._scan_local
        merged: dict[str, dict[str, Any]] = {}
        for _, chunk in _windows(text, size, stride):
            for found in await scan_one(chunk):
                seen = merged.get(found["category"])
                if seen is None or found["confidence"] > seen["confidence"]:
                    merged[found["category"]] = found
        return list(merged.values())

    def check_ready(self) -> None:
        """Verify the provider is usable now; raise if it is not.

        Called by :meth:`PromptiseSecurityScanner.warmup`.  For the local
        provider this asks Ollama whether ``llama-guard3`` is pulled; for
        Azure it checks that the endpoint and key are configured.
        """
        if self.provider == "azure":
            self._azure_credentials()
            return
        import httpx

        try:
            resp = httpx.get(f"{self._OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Content safety needs Ollama at {self._OLLAMA_URL}, which is unreachable: {exc}"
            ) from exc
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(n.split(":")[0] == self._OLLAMA_MODEL for n in names):
            raise RuntimeError(
                f"Ollama is running but {self._OLLAMA_MODEL!r} is not pulled: "
                f"run `ollama pull {self._OLLAMA_MODEL}`"
            )

    async def _scan_local(self, text: str) -> list[dict[str, Any]]:
        """Scan one window using Llama Guard via Ollama."""
        import httpx

        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Task: Check if there is unsafe content in 'User' messages "
            "in conversations according to our safety policy.\n\n"
            "<BEGIN CONVERSATION>\n"
            f"User: {text}\n"
            "<END CONVERSATION>\n\n"
            "Provide your safety assessment for the user message. "
            "First line must read 'safe' or 'unsafe'. "
            "If unsafe, a second line must include a comma-separated list "
            "of violated categories (S1-S13)."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._OLLAMA_URL}/api/generate",
                    json={
                        "model": self._OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 100},
                    },
                )
                resp.raise_for_status()
                response_text = resp.json().get("response", "").strip()
        except httpx.TransportError as exc:
            raise RuntimeError(f"Ollama at {self._OLLAMA_URL} is unreachable: {exc}") from exc

        return self._parse_response(response_text)

    def _azure_credentials(self) -> tuple[str, str]:
        """Return ``(endpoint, key)``, resolving ``${ENV_VAR}`` key syntax."""
        if not self.azure_endpoint or not self.azure_key:
            raise ValueError("azure_endpoint and azure_key required for Azure provider")
        key = self.azure_key
        if key.startswith("${") and key.endswith("}"):
            import os

            var_name = key[2:-1].split(":-")[0]
            key = os.environ.get(var_name, "")
            if not key:
                raise ValueError(f"Environment variable '{var_name}' not set")
        return self.azure_endpoint, key

    async def _scan_azure(self, text: str) -> list[dict[str, Any]]:
        """Scan one window using Azure AI Content Safety API."""
        endpoint, key = self._azure_credentials()
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{endpoint.rstrip('/')}/contentsafety/text:analyze?api-version=2024-09-01",
                json={"text": text},
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Azure returns categoriesAnalysis with severity 0-6
        findings: list[dict[str, Any]] = []
        for cat in data.get("categoriesAnalysis", []):
            severity = cat.get("severity", 0)
            # Normalize Azure severity (0-6) to 0.0-1.0
            confidence = severity / 6.0
            if confidence >= self.threshold:
                findings.append(
                    {
                        "category": cat.get("category", "unknown").lower(),
                        "label": cat.get("category", "unknown"),
                        "confidence": round(confidence, 2),
                    }
                )
        return findings

    def _parse_response(self, text: str) -> list[dict[str, Any]]:
        """Parse Llama Guard response."""
        lines = text.strip().split("\n")
        if not lines or lines[0].strip().lower() == "safe":
            return []

        findings: list[dict[str, Any]] = []
        if len(lines) >= 2:
            cats = [c.strip() for c in lines[1].split(",")]
            for cat_code in cats:
                cat_code = cat_code.upper().strip()
                label = self.CATEGORIES.get(cat_code, cat_code)
                findings.append(
                    {
                        "category": cat_code.lower(),
                        "label": label,
                        "confidence": 0.9,  # Llama Guard is binary, use high confidence
                    }
                )
        else:
            # Just "unsafe" with no categories
            findings.append(
                {
                    "category": "unsafe",
                    "label": "Unsafe content detected",
                    "confidence": 0.9,
                }
            )
        return findings


class NERDetector:
    """Detect unstructured PII using GLiNER zero-shot NER model.

    Finds person names, physical addresses, organizations, and other
    entities that regex cannot reliably detect.

    Args:
        model: HuggingFace model ID or local path.
        labels: Entity types to detect.
        threshold: Confidence threshold (0.0-1.0).
        action: What to do on detection (default: REDACT).

    Example::

        NERDetector()  # default GLiNER PII model
        NERDetector(model="/models/local/gliner", labels=["person", "address"])
    """

    def __init__(
        self,
        *,
        model: str = "knowledgator/gliner-pii-edge-v1.0",
        labels: list[str] | None = None,
        threshold: float = 0.5,
        action: Action = Action.REDACT,
    ) -> None:
        self.model = model
        self.labels = labels or [
            "person",
            "email",
            "phone number",
            "address",
            "date of birth",
            "organization",
            "medical record",
        ]
        self.threshold = threshold
        self.action = action
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load the GLiNER model (cached after first call)."""
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError("gliner required for NER detection: pip install gliner")
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._model = GLiNER.from_pretrained(self.model)
        logger.info("Loaded GLiNER model: %s", self.model)
        return self._model

    async def scan(self, text: str) -> list[dict[str, Any]]:
        """Scan text for named entities.

        Long text is scanned in overlapping windows (GLiNER truncates its
        input), with offsets mapped back to the full text.

        Returns:
            List of dicts with ``text``, ``label``, ``start``, ``end``, ``score``.
        """
        loop = asyncio.get_running_loop()
        model = self._load_model()

        def _predict() -> list[dict[str, Any]]:
            found: dict[tuple[int, int, str], dict[str, Any]] = {}
            for offset, chunk in _windows(text, _NER_WINDOW, _NER_STRIDE):
                for ent in model.predict_entities(chunk, self.labels, threshold=self.threshold):
                    start = offset + ent.get("start", 0)
                    end = offset + ent.get("end", 0)
                    label = ent.get("label", "unknown")
                    found.setdefault(
                        (start, end, label),
                        {
                            "text": ent.get("text", ent.get("word", "")),
                            "label": label,
                            "start": start,
                            "end": end,
                            "score": round(ent.get("score", 0.0), 3),
                        },
                    )
            return list(found.values())

        # GLiNER is CPU-bound — run in executor
        return await loop.run_in_executor(None, _predict)


class CustomRule:
    """A developer-defined regex detection rule.

    Args:
        name: Unique rule identifier.
        pattern: Regex pattern string.
        severity: Finding severity.
        action: What to do on match.
        description: Human-readable description.

    Example::

        CustomRule(
            name="internal_id",
            pattern=r"INT-\\d{8}",
            description="Internal tracking ID",
        )
    """

    def __init__(
        self,
        *,
        name: str,
        pattern: str,
        severity: Severity = Severity.HIGH,
        action: Action = Action.REDACT,
        description: str = "",
    ) -> None:
        self.name = name
        self.compiled = re.compile(pattern)
        self.severity = severity
        self.action = action
        self.description = description or f"Custom rule: {name}"


class PromptiseSecurityScanner:
    """Unified security scanner for agent input and output.

    Compose detection heads to build exactly the scanner you need.
    Each head is a standalone config object — plug in what matters,
    leave out what doesn't.

    **Composable API** (recommended)::

        from promptise.guardrails import (
            PromptiseSecurityScanner,
            InjectionDetector,
            PIIDetector,
            CredentialDetector,
            ContentSafetyDetector,
            NERDetector,
            CustomRule,
        )

        scanner = PromptiseSecurityScanner(
            detectors=[
                InjectionDetector(),
                PIIDetector(categories={PIICategory.CREDIT_CARDS, PIICategory.SSN}),
                CredentialDetector(categories={CredentialCategory.AWS}),
            ],
            custom_rules=[
                CustomRule(name="internal_id", pattern=r"INT-\\d{8}"),
            ],
        )

    **One-liner defaults** (all heads enabled)::

        scanner = PromptiseSecurityScanner.default()

    **Flat API** (backward compatible)::

        scanner = PromptiseSecurityScanner(
            detect_injection=True,
            detect_pii={PIICategory.CREDIT_CARDS, PIICategory.SSN},
            detect_credentials={CredentialCategory.AWS},
        )

    Args:
        detectors: List of detector instances to enable.
        custom_rules: List of :class:`CustomRule` instances.
        fail_open: What to do when an enabled head cannot run (its
            library is missing, its model fails to load, Ollama or Azure
            is unreachable).  ``False`` (the default) fails closed: the
            scan gets a ``BLOCK`` finding, so ``check_input`` /
            ``check_output`` raise :class:`GuardrailViolation`.  ``True``
            logs a warning and lets the text through.  Either way the head
            is reported in :attr:`ScanReport.scanners_skipped`, not in
            ``scanners_run``.
        redact_input: Apply the PII and credential actions to input as
            well as output, and have :meth:`check_input` return the
            redacted text so the agent sends that to the model.  Off by
            default: on input, PII and credentials are warnings and the
            message is passed unchanged.
        scan_tool_results: Scan every tool result before the model sees
            it (indirect prompt injection, leaked secrets).
            ``build_agent`` wraps the agent's tools when this is set; see
            :meth:`check_tool_result`.
        detect_injection: (flat API) Enable injection detection.
        detect_pii: (flat API) Enable PII detection.
        detect_toxicity: (flat API) Enable toxicity detection.
        detect_credentials: (flat API) Enable credential detection.
    """

    # Class-level annotations for attributes set in both init branches.
    # ``None`` means "all groups enabled"; a ``set`` restricts to named groups.
    _pii_groups: set[str] | None
    _cred_groups: set[str] | None

    @classmethod
    def default(cls, **kwargs: Any) -> PromptiseSecurityScanner:
        """Create a scanner with all detection heads enabled (defaults).

        Equivalent to::

            PromptiseSecurityScanner(detectors=[
                InjectionDetector(),
                PIIDetector(),
                CredentialDetector(),
            ])

        Keyword arguments (``fail_open``, ``redact_input``,
        ``scan_tool_results``, ``custom_rules``) are passed through.
        """
        return cls(
            detectors=[
                InjectionDetector(),
                PIIDetector(),
                CredentialDetector(),
            ],
            **kwargs,
        )

    def __init__(
        self,
        *,
        # ── Composable API (recommended) ──
        detectors: list[Any] | None = None,
        custom_rules: list[CustomRule | dict[str, Any]] | None = None,
        fail_open: bool = False,
        redact_input: bool = False,
        scan_tool_results: bool = False,
        # ── Flat API (backward compatible) ──
        detect_injection: bool = True,
        detect_pii: bool | set[PIICategory] = True,
        detect_toxicity: bool = True,
        detect_credentials: bool | set[CredentialCategory] = True,
        injection_model: str = _DEFAULT_INJECTION_MODEL,
        toxicity_model: str = _DEFAULT_TOXICITY_MODEL,
        injection_threshold: float = 0.85,
        toxicity_threshold: float = 0.7,
        on_pii: Action = Action.REDACT,
        on_credentials: Action = Action.REDACT,
        on_toxicity: Action = Action.WARN,
        pii_patterns: list[str] | None = None,
        credential_patterns: list[str] | None = None,
        exclude_patterns: set[str] | None = None,
    ) -> None:
        self.fail_open = fail_open
        self.redact_input = redact_input
        self.scan_tool_results = scan_tool_results

        # Store detectors for warmup() and introspection
        self._detectors: list[Any] = []

        # ── Composable API: detectors list takes priority ──
        if detectors is not None:
            self._detectors = list(detectors)
            # Derive flags from detector types
            self.detect_injection = any(isinstance(d, InjectionDetector) for d in detectors)
            self.detect_pii = any(isinstance(d, PIIDetector) for d in detectors)
            self.detect_credentials = any(isinstance(d, CredentialDetector) for d in detectors)
            self.detect_content_safety = any(
                isinstance(d, ContentSafetyDetector) for d in detectors
            )
            self.detect_ner = any(isinstance(d, NERDetector) for d in detectors)
            # If ContentSafetyDetector is present, it replaces basic toxicity
            self.detect_toxicity = (
                (
                    not self.detect_content_safety
                    and any(isinstance(d, InjectionDetector) for d in detectors)
                )
                if False
                else False
            )  # disabled when composable API used

            # Store detector instances for scan methods
            self._content_safety_det: ContentSafetyDetector | None = next(
                (d for d in detectors if isinstance(d, ContentSafetyDetector)), None
            )
            self._ner_det: NERDetector | None = next(
                (d for d in detectors if isinstance(d, NERDetector)), None
            )

            # Extract config from detector instances
            inj = next((d for d in detectors if isinstance(d, InjectionDetector)), None)
            self._injection_model_name = inj.model if inj else injection_model
            self._injection_threshold = inj.threshold if inj else injection_threshold

            self._toxicity_model_name = toxicity_model
            self._toxicity_threshold = toxicity_threshold

            pii_det = next((d for d in detectors if isinstance(d, PIIDetector)), None)
            self._on_pii = pii_det.action if pii_det else on_pii
            self._exclude = pii_det.exclude if pii_det else set()

            cred_det = next((d for d in detectors if isinstance(d, CredentialDetector)), None)
            self._on_credentials = cred_det.action if cred_det else on_credentials
            self._on_toxicity = on_toxicity

            # PII groups from detector
            if pii_det:
                self._pii_groups = pii_det.groups
            else:
                self._pii_groups = set()
            self._pii_name_include = None

            # Credential groups from detector
            if cred_det:
                self._cred_groups = cred_det.groups
                cred_exclude = cred_det.exclude
            else:
                self._cred_groups = set()
                cred_exclude = set()
            self._cred_name_include = None
            if cred_exclude:
                self._exclude = self._exclude | cred_exclude

        else:
            # ── Flat API (backward compatible) ──
            self.detect_injection = detect_injection
            self.detect_toxicity = detect_toxicity
            self.detect_content_safety = False
            self.detect_ner = False
            self._content_safety_det = None
            self._ner_det = None

            self._injection_model_name = injection_model
            self._toxicity_model_name = toxicity_model
            self._injection_threshold = injection_threshold
            self._toxicity_threshold = toxicity_threshold
            self._on_pii = on_pii
            self._on_credentials = on_credentials
            self._on_toxicity = on_toxicity
            self._exclude = exclude_patterns or set()

            # ── PII filtering: bool, set[PIICategory], or name list ──
            if isinstance(detect_pii, set):
                self.detect_pii = True
                if PIICategory.ALL in detect_pii:
                    self._pii_groups = None
                else:
                    self._pii_groups = {c.value for c in detect_pii}
            elif detect_pii:
                self.detect_pii = True
                self._pii_groups = None
            else:
                self.detect_pii = False
                self._pii_groups = set()
            self._pii_name_include = set(pii_patterns) if pii_patterns else None

            # ── Credential filtering ──
            if isinstance(detect_credentials, set):
                self.detect_credentials = True
                if CredentialCategory.ALL in detect_credentials:
                    self._cred_groups = None
                else:
                    self._cred_groups = {c.value for c in detect_credentials}
            elif detect_credentials:
                self.detect_credentials = True
                self._cred_groups = None
            else:
                self.detect_credentials = False
                self._cred_groups = set()
            self._cred_name_include = set(credential_patterns) if credential_patterns else None

        # Custom rules: support both CustomRule objects and dicts
        self._custom_rules: list[tuple[str, str, re.Pattern[str], Severity, str, Action]] = []
        for rule in custom_rules or []:
            if isinstance(rule, CustomRule):
                self._custom_rules.append(
                    (
                        rule.name,
                        rule.name,
                        rule.compiled,
                        rule.severity,
                        rule.description,
                        rule.action,
                    )
                )
            else:
                self._custom_rules.append(
                    (
                        rule["name"],
                        rule.get("category", rule["name"]),
                        re.compile(rule["pattern"]),
                        Severity(rule.get("severity", "high")),
                        rule.get("description", f"Custom rule: {rule['name']}"),
                        Action(rule.get("action", "redact")),
                    )
                )

    # ── Guard protocol ────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load ML models so the first scan is fast.

        Call this at startup to avoid download/load latency on the
        first message, and to fail fast: a missing library or model, or an
        unreachable Ollama for :class:`ContentSafetyDetector`, raises here
        (whatever ``fail_open`` says) instead of surfacing on the first
        request.  Safe to call multiple times (models are cached).

        Example::

            scanner = PromptiseSecurityScanner()
            scanner.warmup()  # downloads + loads models NOW
            agent = await build_agent(..., guardrails=scanner)
        """
        if self.detect_injection:
            _load_classifier(self._injection_model_name)
            logger.info("Warmed up injection model: %s", self._injection_model_name)
        if self.detect_toxicity:
            _load_classifier(self._toxicity_model_name)
            logger.info("Warmed up toxicity model: %s", self._toxicity_model_name)
        if self.detect_ner and self._ner_det is not None:
            self._ner_det._load_model()
            logger.info("Warmed up NER model: %s", self._ner_det.model)
        if self.detect_content_safety and self._content_safety_det is not None:
            self._content_safety_det.check_ready()
            logger.info(
                "Content safety detector ready (provider: %s)",
                self._content_safety_det.provider,
            )

    async def check_input(self, text: str) -> str:
        """Scan input text.  Raises :class:`GuardrailViolation` on block.

        Called by the agent before any processing (memory, tools, LLM).

        Returns:
            The text to send on: the redacted text when ``redact_input``
            is set and something was redacted, otherwise *text* unchanged.
            The agent replaces the user's message with it.
        """
        if isinstance(text, dict):
            # Extract message text from LangChain input format
            msgs = text.get("messages", [])
            if msgs:
                last = msgs[-1]
                text = last.get("content", "") if isinstance(last, dict) else str(last)
            else:
                text = str(text)
        if not isinstance(text, str):
            text = str(text)  # e.g. a list of multimodal content blocks

        report = await self.scan_text(text, direction="input")
        if not report.passed:
            raise GuardrailViolation(report, direction="input")
        if self.redact_input and report.redacted_text is not None:
            return report.redacted_text
        return text

    async def check_output(self, output: Any) -> Any:
        """Scan output text.  Redacts PII/credentials.  Blocks on injection.

        Called by the agent after the LLM response, before returning.
        """
        text = str(output) if not isinstance(output, str) else output
        report = await self.scan_text(text, direction="output")

        if not report.passed:
            raise GuardrailViolation(report, direction="output")

        # Apply redactions if any
        if report.redacted_text and report.redacted_text != text:
            return report.redacted_text
        return output

    async def check_tool_result(self, tool_name: str, result: str) -> str:
        """Scan a tool result before the model reads it.

        A tool result is where indirect prompt injection arrives (a web
        page, an email, a ticket), so every enabled head runs, the
        injection model included.  PII and credentials get their
        configured actions, as on output, so a secret a tool returns is
        redacted before the model (or ``result["messages"]``) holds it.

        ``build_agent`` calls this for every tool call when
        ``scan_tool_results=True``; a blocked result is replaced with a
        short notice the model can see, and the raw result is dropped.

        Raises:
            GuardrailViolation: With ``direction="tool"`` on a block.
        """
        report = await self.scan_text(result, direction="tool")
        if not report.passed:
            logger.warning(
                "Guardrails blocked the result of tool %r: %s",
                tool_name,
                "; ".join(f.description for f in report.blocked[:3]),
            )
            raise GuardrailViolation(report, direction="tool")
        if report.redacted_text is not None:
            return report.redacted_text
        return result

    # ── Core scan ─────────────────────────────────────────────────────

    async def scan_text(
        self,
        text: str,
        *,
        direction: str = "input",
    ) -> ScanReport:
        """Run all enabled detection heads on the given text.

        Args:
            text: Text to scan.
            direction: ``"input"``, ``"output"`` or ``"tool"`` (a tool
                result).  On input, PII and credential findings are
                warnings unless ``redact_input`` is set; on output and
                tool results they get their configured action.  The
                injection head skips output.

        Returns:
            A :class:`ScanReport` with all findings.  An enabled head
            that could not run is listed in ``scanners_skipped`` and,
            unless ``fail_open`` is set, adds a ``BLOCK`` finding.
        """
        start_time = time.perf_counter()
        findings: list[SecurityFinding] = []
        scanners_run: list[str] = []
        scanners_skipped: dict[str, str] = {}

        # Regex heads (sync, cannot fail to run)
        if self.detect_pii:
            scanners_run.append("pii")
            findings.extend(self._scan_pii(text, direction))

        if self.detect_credentials:
            scanners_run.append("credential")
            findings.extend(self._scan_credentials(text, direction))

        # Model and service heads: each one can fail to run
        heads: list[tuple[str, Any]] = []
        if self.detect_injection and direction != "output":
            heads.append(("injection", self._scan_injection))
        if self.detect_toxicity:
            heads.append(("toxicity", self._scan_toxicity))
        if self.detect_content_safety and self._content_safety_det is not None:
            heads.append(("content_safety", self._scan_content_safety))
        if self.detect_ner and self._ner_det is not None:
            heads.append(("ner", self._scan_ner))

        for name, head in heads:
            try:
                head_findings = await head(text, direction)
            except Exception as exc:
                reason = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
                scanners_skipped[name] = reason
                findings.extend(self._head_unavailable(name, reason))
                continue
            scanners_run.append(name)
            findings.extend(head_findings)

        # Custom rules always run if defined
        if self._custom_rules:
            scanners_run.append("custom")
            findings.extend(self._scan_custom(text))

        # Build redacted text for output direction
        redacted_text = None
        redact_findings = [f for f in findings if f.action == Action.REDACT]
        if redact_findings:
            redacted_text = self._apply_redactions(text, redact_findings)

        passed = not any(f.action == Action.BLOCK for f in findings)
        duration = (time.perf_counter() - start_time) * 1000

        # Attach caller identity so audit logs can attribute findings to a
        # specific tenant even after the contextvar is reset.  Done here —
        # not in the constructor — so standalone users of ``ScanReport``
        # aren't forced through the CallerContext import.
        caller_user_id: str | None = None
        caller_session_id: str | None = None
        caller_roles: tuple[str, ...] = ()
        try:
            from .agent import get_current_caller
        except Exception:  # pragma: no cover — defensive
            get_current_caller = None  # type: ignore[assignment]
        if get_current_caller is not None:
            caller = get_current_caller()
            if caller is not None:
                caller_user_id = getattr(caller, "user_id", None)
                meta = getattr(caller, "metadata", None) or {}
                caller_session_id = meta.get("session_id")
                roles = getattr(caller, "roles", None) or ()
                caller_roles = tuple(sorted(str(r) for r in roles))

        return ScanReport(
            passed=passed,
            findings=findings,
            duration_ms=round(duration, 2),
            scanners_run=scanners_run,
            text_length=len(text),
            redacted_text=redacted_text,
            user_id=caller_user_id,
            session_id=caller_session_id,
            caller_roles=caller_roles,
            scanners_skipped=scanners_skipped,
        )

    def _head_unavailable(self, name: str, reason: str) -> list[SecurityFinding]:
        """Handle a head that could not run: fail closed unless ``fail_open``."""
        if self.fail_open:
            logger.warning("Guardrail head %r skipped (fail_open=True): %s", name, reason)
            return []
        logger.error("Guardrail head %r could not run, blocking (fail-closed): %s", name, reason)
        return [
            SecurityFinding(
                detector=name,
                category="scanner_unavailable",
                severity=Severity.CRITICAL,
                confidence=1.0,
                matched_text="",
                start=0,
                end=0,
                action=Action.BLOCK,
                description=(
                    f"Guardrail head {name!r} could not run: {reason} "
                    "(blocked because fail_open=False)"
                ),
                metadata={"reason": reason, "fail_open": False},
            )
        ]

    # ── Detection heads ───────────────────────────────────────────────

    def _scan_pii(self, text: str, direction: str) -> list[SecurityFinding]:
        """Regex + Luhn validation for PII detection."""
        findings: list[SecurityFinding] = []
        action = self._on_pii if direction != "input" or self.redact_input else Action.WARN

        for name, category, pattern, severity, desc, group in _PII_PATTERNS:
            # Exclude blacklisted patterns
            if name in self._exclude:
                continue
            # Group-based filtering (PIICategory enum)
            if self._pii_groups is not None and group not in self._pii_groups:
                continue
            # Legacy name-based filtering
            if self._pii_name_include and name not in self._pii_name_include:
                continue

            for m in pattern.finditer(text):
                matched = m.group()

                # Credit card patterns require Luhn validation
                if "credit_card" in category:
                    digits_only = re.sub(r"[\s\-]", "", matched)
                    if not _luhn_check(digits_only):
                        continue

                findings.append(
                    SecurityFinding(
                        detector="pii",
                        category=category,
                        severity=severity,
                        confidence=1.0,
                        matched_text=matched,
                        start=m.start(),
                        end=m.end(),
                        action=action,
                        description=f"{desc} detected",
                        metadata={"pattern": name, "luhn_valid": "credit_card" in category},
                    )
                )
        return findings

    def _scan_credentials(self, text: str, direction: str) -> list[SecurityFinding]:
        """Regex patterns for credential/secret detection."""
        findings: list[SecurityFinding] = []
        action = self._on_credentials if direction != "input" or self.redact_input else Action.WARN

        for name, category, pattern, severity, desc, group in _CRED_PATTERNS:
            if name in self._exclude:
                continue
            # Group-based filtering (CredentialCategory enum)
            if self._cred_groups is not None and group not in self._cred_groups:
                continue
            # Legacy name-based filtering
            if self._cred_name_include and name not in self._cred_name_include:
                continue

            for m in pattern.finditer(text):
                findings.append(
                    SecurityFinding(
                        detector="credential",
                        category=category,
                        severity=severity,
                        confidence=1.0,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        action=action,
                        description=f"{desc} detected",
                        metadata={"pattern": name},
                    )
                )
        return findings

    async def _classify(self, model_name: str, text: str) -> list[tuple[int, str, str, float]]:
        """Run a text classifier over overlapping windows of *text*.

        Returns ``(offset, window, label, score)`` per window.  Raises if
        the model cannot be loaded or run; the caller decides whether that
        fails open or closed.
        """
        pipe = _load_classifier(model_name)
        windows = _windows(text, _CLASSIFIER_WINDOW, _CLASSIFIER_STRIDE)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, pipe, [chunk for _, chunk in windows])
        out: list[tuple[int, str, str, float]] = []
        for (offset, chunk), item in zip(windows, results, strict=True):
            top = _top_prediction(item)
            out.append((offset, chunk, str(top.get("label", "")), float(top.get("score", 0.0))))
        return out

    async def _scan_injection(self, text: str, direction: str) -> list[SecurityFinding]:
        """Prompt injection detection via local DeBERTa model.

        No regex pre-filter — the model handles all classification to
        avoid false positives on benign phrases like "pretend to be".
        The whole text is classified in overlapping windows, so padding
        cannot push an attack past the model's input limit.
        """
        # Agent replies aren't an injection vector; user input and tool
        # results are.
        if direction == "output" or not text.strip():
            return []

        windows = await self._classify(self._injection_model_name, text)
        hits = [
            (score, offset, chunk, label.upper())
            for offset, chunk, label, score in windows
            # protectai model: LABEL_1 = injection, LABEL_0 = benign
            if label.upper() in ("INJECTION", "LABEL_1", "1") and score >= self._injection_threshold
        ]
        if not hits:
            return []
        score, offset, chunk, label = max(hits, key=lambda h: h[0])
        return [
            SecurityFinding(
                detector="injection",
                category="prompt_injection_model",
                severity=Severity.CRITICAL,
                confidence=score,
                matched_text=chunk[:100] + ("..." if len(chunk) > 100 else ""),
                start=offset,
                end=offset + len(chunk),
                action=Action.BLOCK,
                description=f"Prompt injection detected by model (confidence: {score:.2%})",
                metadata={
                    "method": "model",
                    "model": self._injection_model_name,
                    "label": label,
                    "score": score,
                    "windows": len(windows),
                },
            )
        ]

    async def _scan_toxicity(self, text: str, direction: str) -> list[SecurityFinding]:
        """Toxicity detection via local transformer model (windowed)."""
        if not text.strip():
            return []
        windows = await self._classify(self._toxicity_model_name, text)
        hits = [
            (score, offset, chunk, label.lower())
            for offset, chunk, label, score in windows
            if label.lower() in ("toxic", "label_1", "1") and score >= self._toxicity_threshold
        ]
        if not hits:
            return []
        score, offset, chunk, label = max(hits, key=lambda h: h[0])
        return [
            SecurityFinding(
                detector="toxicity",
                category=f"toxic_{label}",
                severity=Severity.HIGH if score > 0.9 else Severity.MEDIUM,
                confidence=score,
                matched_text=chunk[:100] + ("..." if len(chunk) > 100 else ""),
                start=offset,
                end=offset + len(chunk),
                action=self._on_toxicity,
                description=f"Toxic content detected (confidence: {score:.2%})",
                metadata={
                    "method": "model",
                    "model": self._toxicity_model_name,
                    "label": label,
                    "score": score,
                    "windows": len(windows),
                },
            )
        ]

    async def _scan_content_safety(self, text: str, direction: str) -> list[SecurityFinding]:
        """Content safety via Llama Guard (local) or Azure AI (cloud)."""
        det = self._content_safety_det
        if det is None or not text.strip():
            return []
        return [
            SecurityFinding(
                detector="content_safety",
                category=r.get("category", "unsafe"),
                severity=Severity.HIGH,
                confidence=r.get("confidence", 0.9),
                matched_text=text[:100] + ("..." if len(text) > 100 else ""),
                start=0,
                end=len(text),
                action=det.action,
                description=f"Content safety violation: {r.get('label', 'unsafe')}",
                metadata={
                    "method": "model",
                    "provider": det.provider,
                    "category_code": r.get("category"),
                },
            )
            for r in await det.scan(text)
        ]

    async def _scan_ner(self, text: str, direction: str) -> list[SecurityFinding]:
        """Named Entity Recognition via GLiNER."""
        det = self._ner_det
        if det is None or not text.strip():
            return []
        findings: list[SecurityFinding] = []
        for ent in await det.scan(text):
            label = ent.get("label", "entity")
            matched = ent.get("text", "")
            findings.append(
                SecurityFinding(
                    detector="ner",
                    category=f"ner_{label.replace(' ', '_').lower()}",
                    severity=Severity.MEDIUM,
                    confidence=ent.get("score", 0.5),
                    matched_text=matched,
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                    action=det.action,
                    description=f"{label} detected: '{matched}'",
                    metadata={
                        "method": "model",
                        "model": det.model,
                        "entity_type": label,
                    },
                )
            )
        return findings

    # ── Redaction engine ──────────────────────────────────────────────

    def _scan_custom(self, text: str) -> list[SecurityFinding]:
        """Run developer-defined custom regex rules."""
        findings: list[SecurityFinding] = []
        for name, category, pattern, severity, desc, action in self._custom_rules:
            for m in pattern.finditer(text):
                findings.append(
                    SecurityFinding(
                        detector="custom",
                        category=category,
                        severity=severity,
                        confidence=1.0,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        action=action,
                        description=desc,
                        metadata={"pattern": name, "custom": True},
                    )
                )
        return findings

    # ── Utility: list available patterns ──────────────────────────────

    @staticmethod
    def list_pii_patterns() -> list[str]:
        """Return names of all built-in PII patterns."""
        return [name for name, *_ in _PII_PATTERNS]

    @staticmethod
    def list_credential_patterns() -> list[str]:
        """Return names of all built-in credential patterns."""
        return [name for name, *_ in _CRED_PATTERNS]

    # ── Redaction engine ──────────────────────────────────────────────

    # When overlapping spans tie on length, the more specific detector names it.
    _REDACTION_PRIORITY = {"custom": 3, "credential": 2, "pii": 1, "ner": 0}

    @staticmethod
    def _apply_redactions(text: str, findings: list[SecurityFinding]) -> str:
        """Replace detected spans with redaction labels.

        Overlapping spans (a connection string whose ``user:pass@host``
        also reads as an email, two phone patterns on one number) are
        merged into one span covering all of them and replaced once,
        labelled by the longest finding, then the most specific detector.
        Replacing them one by one would cut into text after the first
        replacement.  Spans are replaced right to left so offsets stay
        valid.
        """
        priority = PromptiseSecurityScanner._REDACTION_PRIORITY
        groups: list[tuple[int, int, SecurityFinding]] = []  # (start, end, label finding)
        for f in sorted(findings, key=lambda f: (f.start, -f.end)):
            if f.end <= f.start:
                continue
            if groups and f.start < groups[-1][1]:
                start, end, best = groups[-1]
                if (f.end - f.start, priority.get(f.detector, 0)) > (
                    best.end - best.start,
                    priority.get(best.detector, 0),
                ):
                    best = f
                groups[-1] = (start, max(end, f.end), best)
            else:
                groups.append((f.start, f.end, f))
        result = text
        for start, end, best in reversed(groups):
            result = result[:start] + f"[{best.category.upper()}]" + result[end:]
        return result


# ═══════════════════════════════════════════════════════════════════════
# Tool-result scanning (indirect prompt injection)
# ═══════════════════════════════════════════════════════════════════════


class _GuardedTool(BaseTool):
    """Wraps a tool so its result passes the guardrails before the model sees it.

    Transparent to the LLM — same name, description, and schema as the
    inner tool.  A blocked result is replaced with a short notice; a
    redacted one is returned redacted.  Either way the raw result never
    reaches the model or the agent's message history.
    """

    _inner: BaseTool = PrivateAttr()
    _guard: Any = PrivateAttr()
    _event_notifier: Any = PrivateAttr(default=None)

    def __init__(self, inner: BaseTool, guard: Any, event_notifier: Any = None) -> None:
        super().__init__(
            name=inner.name,
            description=inner.description,
            args_schema=getattr(inner, "args_schema", None),
            return_direct=inner.return_direct,
        )
        self._inner = inner
        self._guard = guard
        self._event_notifier = event_notifier

    async def _arun(self, *args: Any, promptise_guard_config: RunnableConfig, **kwargs: Any) -> Any:
        # LangChain fills ``promptise_guard_config`` by its type (an unusual
        # name, so it cannot shadow a tool argument).  No callbacks on the
        # inner call: observability records this wrapper's run, with the
        # scanned output, once.
        tool_input: Any = args[0] if len(args) == 1 and not kwargs else kwargs
        result = await self._inner.ainvoke(
            tool_input,
            config=patch_config(promptise_guard_config, callbacks=CallbackManager(handlers=[])),
        )
        text = result if isinstance(result, str) else str(result)
        try:
            checked = await self._guard.check_tool_result(self.name, text)
        except GuardrailViolation as violation:
            self._emit("guardrail.blocked", "warning")
            details = "; ".join(f.description for f in violation.report.blocked[:3])
            return f"[Tool result withheld by guardrails: {details}]"
        if checked == text:
            return result
        self._emit("guardrail.redacted", "info")
        return checked

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("Guardrail-wrapped tools are async-only; use ainvoke().")

    def _emit(self, event_type: str, severity: str) -> None:
        if self._event_notifier is None:
            return
        from .events import emit_event

        emit_event(
            self._event_notifier,
            event_type,
            severity,
            {"direction": "tool", "tool": self.name},
        )


def wrap_tools_with_guardrails(
    tools: list[BaseTool],
    guard: Any,
    *,
    event_notifier: Any = None,
) -> list[BaseTool]:
    """Wrap every tool so its result is scanned by ``guard.check_tool_result``.

    ``build_agent`` calls this when the guardrails object has
    ``scan_tool_results=True``.  Order is preserved.
    """
    return [_GuardedTool(tool, guard, event_notifier) for tool in tools]
