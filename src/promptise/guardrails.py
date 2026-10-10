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
        direction: Whether this was an input or output scan.
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
# Blood type — contextual: a bare "A-" or "B+" is far more often an ID prefix
# ("A-1001") or a grade than a blood type, so require "blood type/group".
_pii(
    "blood_type",
    "medical",
    r"(?i)\bblood[\s_-]*(?:type|group)\s*(?:is\s+)?[:=]?\s*(?:AB|A|B|O)\s?"
    r"(?:[+-]|(?:Rh\s*)?(?:pos|neg)(?:itive|ative)?\b)(?![\w+-])",
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

    async def scan(self, text: str) -> list[dict[str, Any]]:
        """Scan text for content safety violations.

        Returns:
            List of dicts with ``category``, ``label``, ``confidence``.
        """
        if self.provider == "azure":
            return await self._scan_azure(text)
        return await self._scan_local(text)

    async def _scan_local(self, text: str) -> list[dict[str, Any]]:
        """Scan using Llama Guard via Ollama."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for local content safety: pip install httpx")

        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Task: Check if there is unsafe content in 'User' messages "
            "in conversations according to our safety policy.\n\n"
            "<BEGIN CONVERSATION>\n"
            f"User: {text[:4000]}\n"
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
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama-guard3",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 100},
                    },
                )
                resp.raise_for_status()
                response_text = resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("Content safety local scan failed: %s", exc)
            return []

        return self._parse_response(response_text)

    async def _scan_azure(self, text: str) -> list[dict[str, Any]]:
        """Scan using Azure AI Content Safety API."""
        if not self.azure_endpoint or not self.azure_key:
            raise ValueError("azure_endpoint and azure_key required for Azure provider")

        # Resolve env var syntax
        key = self.azure_key
        if key.startswith("${") and key.endswith("}"):
            import os

            var_name = key[2:-1].split(":-")[0]
            key = os.environ.get(var_name, "")
            if not key:
                raise ValueError(f"Environment variable '{var_name}' not set")

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for Azure content safety: pip install httpx")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.azure_endpoint.rstrip('/')}/contentsafety/text:analyze?api-version=2024-09-01",
                    json={"text": text[:10000]},
                    headers={
                        "Ocp-Apim-Subscription-Key": key,
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Content safety Azure scan failed: %s", exc)
            return []

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

        Returns:
            List of dicts with ``text``, ``label``, ``start``, ``end``, ``score``.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        model = self._load_model()

        # GLiNER is CPU-bound — run in executor
        entities = await loop.run_in_executor(
            None,
            lambda: model.predict_entities(text[:5000], self.labels, threshold=self.threshold),
        )

        return [
            {
                "text": ent.get("text", ent.get("word", "")),
                "label": ent.get("label", "unknown"),
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
                "score": round(ent.get("score", 0.0), 3),
            }
            for ent in entities
        ]


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
    def default(cls) -> PromptiseSecurityScanner:
        """Create a scanner with all detection heads enabled (defaults).

        Equivalent to::

            PromptiseSecurityScanner(detectors=[
                InjectionDetector(),
                PIIDetector(),
                CredentialDetector(),
            ])
        """
        return cls(
            detectors=[
                InjectionDetector(),
                PIIDetector(),
                CredentialDetector(),
            ]
        )

    def __init__(
        self,
        *,
        # ── Composable API (recommended) ──
        detectors: list[Any] | None = None,
        custom_rules: list[CustomRule | dict[str, Any]] | None = None,
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
        first message.  Safe to call multiple times (models are cached).

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
            logger.info(
                "Content safety detector ready (provider: %s)",
                self._content_safety_det.provider,
            )

    async def check_input(self, text: str) -> str:
        """Scan input text.  Raises :class:`GuardrailViolation` on block.

        Called by the agent before any processing (memory, tools, LLM).
        """
        if isinstance(text, dict):
            # Extract message text from LangChain input format
            msgs = text.get("messages", [])
            if msgs:
                last = msgs[-1]
                text = last.get("content", "") if isinstance(last, dict) else str(last)
            else:
                text = str(text)

        report = await self.scan_text(text, direction="input")
        if not report.passed:
            raise GuardrailViolation(report, direction="input")
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
            direction: ``"input"`` or ``"output"`` — affects default actions.

        Returns:
            A :class:`ScanReport` with all findings.
        """
        start_time = time.monotonic()
        findings: list[SecurityFinding] = []
        scanners_run: list[str] = []

        # Run all detection heads (regex heads are sync, model heads async)
        if self.detect_pii:
            scanners_run.append("pii")
            findings.extend(self._scan_pii(text, direction))

        if self.detect_credentials:
            scanners_run.append("credential")
            findings.extend(self._scan_credentials(text, direction))

        if self.detect_injection:
            scanners_run.append("injection")
            findings.extend(await self._scan_injection(text, direction))

        if self.detect_toxicity:
            scanners_run.append("toxicity")
            findings.extend(await self._scan_toxicity(text, direction))

        if self.detect_content_safety and self._content_safety_det is not None:
            scanners_run.append("content_safety")
            findings.extend(await self._scan_content_safety(text, direction))

        if self.detect_ner and self._ner_det is not None:
            scanners_run.append("ner")
            findings.extend(await self._scan_ner(text, direction))

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
        duration = (time.monotonic() - start_time) * 1000

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
        )

    # ── Detection heads ───────────────────────────────────────────────

    def _scan_pii(self, text: str, direction: str) -> list[SecurityFinding]:
        """Regex + Luhn validation for PII detection."""
        findings: list[SecurityFinding] = []
        action = self._on_pii if direction == "output" else Action.WARN

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
        action = self._on_credentials if direction == "output" else Action.WARN

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

    async def _scan_injection(self, text: str, direction: str) -> list[SecurityFinding]:
        """Prompt injection detection via local DeBERTa model.

        No regex pre-filter — the model handles all classification to
        avoid false positives on benign phrases like "pretend to be".
        """
        findings: list[SecurityFinding] = []

        # Only scan input for injection — output isn't an injection risk
        if direction != "input":
            return findings

        # Model-based classification
        try:
            pipe = _load_classifier(self._injection_model_name)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, pipe, text[:512])
            if result and len(result) > 0:
                label = result[0].get("label", "").upper()
                score = result[0].get("score", 0.0)

                # protectai model: LABEL_1 = injection, LABEL_0 = benign
                is_injection = label in ("INJECTION", "LABEL_1", "1")
                if is_injection and score >= self._injection_threshold:
                    findings.append(
                        SecurityFinding(
                            detector="injection",
                            category="prompt_injection_model",
                            severity=Severity.CRITICAL,
                            confidence=score,
                            matched_text=text[:100] + ("..." if len(text) > 100 else ""),
                            start=0,
                            end=len(text),
                            action=Action.BLOCK,
                            description=f"Prompt injection detected by model (confidence: {score:.2%})",
                            metadata={
                                "method": "model",
                                "model": self._injection_model_name,
                                "label": label,
                                "score": score,
                            },
                        )
                    )
        except ImportError:
            logger.warning("transformers not installed — skipping ML injection detection")
        except Exception as exc:
            logger.warning("Injection model error (scan continues): %s", exc)

        return findings

    async def _scan_toxicity(self, text: str, direction: str) -> list[SecurityFinding]:
        """Toxicity detection via local transformer model."""
        findings: list[SecurityFinding] = []

        try:
            pipe = _load_classifier(self._toxicity_model_name)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, pipe, text[:512])
            if result and len(result) > 0:
                label = result[0].get("label", "").lower()
                score = result[0].get("score", 0.0)

                is_toxic = label in ("toxic", "label_1", "1")
                if is_toxic and score >= self._toxicity_threshold:
                    findings.append(
                        SecurityFinding(
                            detector="toxicity",
                            category=f"toxic_{label}",
                            severity=Severity.HIGH if score > 0.9 else Severity.MEDIUM,
                            confidence=score,
                            matched_text=text[:100] + ("..." if len(text) > 100 else ""),
                            start=0,
                            end=len(text),
                            action=self._on_toxicity,
                            description=f"Toxic content detected (confidence: {score:.2%})",
                            metadata={
                                "method": "model",
                                "model": self._toxicity_model_name,
                                "label": label,
                                "score": score,
                            },
                        )
                    )
        except ImportError:
            logger.warning("transformers not installed — skipping ML toxicity detection")
        except Exception as exc:
            logger.warning("Toxicity model error (scan continues): %s", exc)

        return findings

    async def _scan_content_safety(self, text: str, direction: str) -> list[SecurityFinding]:
        """Content safety via Llama Guard (local) or Azure AI (cloud)."""
        findings: list[SecurityFinding] = []
        det = self._content_safety_det
        if det is None:
            return findings

        try:
            results = await det.scan(text)
            for r in results:
                findings.append(
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
                )
        except ImportError:
            logger.warning("httpx not installed — skipping content safety scan")
        except Exception as exc:
            logger.warning("Content safety scan error (continues): %s", exc)

        return findings

    async def _scan_ner(self, text: str, direction: str) -> list[SecurityFinding]:
        """Named Entity Recognition via GLiNER."""
        findings: list[SecurityFinding] = []
        det = self._ner_det
        if det is None:
            return findings

        try:
            entities = await det.scan(text)
            for ent in entities:
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
        except ImportError:
            logger.warning("gliner not installed — skipping NER detection")
        except Exception as exc:
            logger.warning("NER scan error (continues): %s", exc)

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

    @staticmethod
    def _apply_redactions(text: str, findings: list[SecurityFinding]) -> str:
        """Replace detected spans with redaction labels.

        Overlapping spans (two patterns matching the same phone number, say)
        are merged first and replaced once, labelled by the earliest and
        longest finding; replacing each one separately would cut into the
        text after the first replacement.  Spans are then replaced right to
        left so character offsets remain valid.
        """
        spans: list[list[Any]] = []  # [start, end, label]
        for f in sorted(findings, key=lambda f: (f.start, -f.end)):
            if spans and f.start < spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], f.end)
            else:
                spans.append([f.start, f.end, f"[{f.category.upper()}]"])
        result = text
        for start, end, label in reversed(spans):
            result = result[:start] + label + result[end:]
        return result
