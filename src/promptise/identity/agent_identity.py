"""The :class:`AgentIdentity` public class — who an agent is.

Agent Identity answers a single question: *which agent took this
action?* Every agent gets a stable, traceable identity so its tool
calls, audit entries, and outbound requests can be attributed to it.

An identity comes in two tiers:

* **Local** — just an ``agent_id`` (plus optional name/owner/labels).
  Needs no infrastructure. It is the value the framework stamps onto
  observability events and audit logs so you can see which agent did
  what.
* **Verifiable** — additionally backed by a *credential provider*
  (Entra, AWS, GCP, SPIFFE, or a generic OIDC issuer) that mints a
  signed JWT proving the identity. The agent presents this credential
  to the resources it calls — an MCP server, an HTTP API — so they can
  authenticate and attribute the caller cryptographically rather than
  taking a self-asserted id on trust.

Typical use::

    from promptise.identity import AgentIdentity

    # Local identity — zero infrastructure.
    identity = AgentIdentity("billing-bot", name="Billing Bot")

    # Verifiable identity — Entra Agent ID backs it so MCP servers can
    # verify the caller really is billing-bot.
    identity = AgentIdentity.from_entra(
        "billing-bot", client_id="...", resource="api://my-mcp-server"
    )

    agent = await build_agent(model=..., servers={...}, identity=identity)
    # every tool call / LLM turn / audit entry is now attributed to it
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from ._core.cache import decode_jwt_claims
from ._core.errors import CredentialAcquisitionError, ProviderConfigError
from ._core.provider import IdentityProvider
from ._internal.detect import detect_platform
from .providers.aws import _DEFAULT_AUDIENCE as _AWS_DEFAULT_AUDIENCE
from .providers.aws import _DEFAULT_SIGNING_ALGORITHM as _AWS_DEFAULT_SIGNING_ALGORITHM
from .providers.aws import from_aws as _from_aws
from .providers.entra import _DEFAULT_RESOURCE as _ENTRA_DEFAULT_RESOURCE
from .providers.entra import from_entra as _from_entra
from .providers.gcp import _DEFAULT_AUDIENCE as _GCP_DEFAULT_AUDIENCE
from .providers.gcp import (
    _DEFAULT_SERVICE_ACCOUNT_EMAIL as _GCP_DEFAULT_SERVICE_ACCOUNT_EMAIL,
)
from .providers.gcp import from_gcp as _from_gcp
from .providers.oidc import from_oidc as _from_oidc
from .providers.spiffe import _DEFAULT_AUDIENCE as _SPIFFE_DEFAULT_AUDIENCE
from .providers.spiffe import from_spiffe as _from_spiffe


class AgentIdentity:
    """The identity of an agent — for tracing and attributing its actions.

    Construct a local identity directly, or use a credential factory
    (:meth:`from_entra`, :meth:`from_aws`, :meth:`from_gcp`,
    :meth:`from_spiffe`, :meth:`from_oidc`, or :meth:`auto`) to make it
    verifiable. An instance is immutable in identity and safe to share
    across threads.

    Args:
        agent_id: Optional local handle. For a **verifiable** identity you
            can omit it: the authoritative identifier then comes from the
            IdP — the ``sub`` (or ``oid``) claim of the credential — via
            :meth:`subject` and :meth:`resolve_identifier`. A **local**
            (unbacked) identity must supply one, since there is no IdP to
            ask.
        name: Optional human-readable display name.
        owner: Optional owner — the team or person operating the agent.
        labels: Optional free-form key/value metadata (team, environment,
            version) carried alongside the identity on every trace.
        credential: Optional :class:`IdentityProvider` that makes the
            identity verifiable. Prefer the factory classmethods; pass
            one directly only when wrapping a custom provider.

    Raises:
        ProviderConfigError: If neither an ``agent_id`` nor a
            ``credential`` is supplied — the identity would have no
            identifier and no IdP to derive one from.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        credential: IdentityProvider | None = None,
    ) -> None:
        normalized = agent_id.strip() if agent_id and agent_id.strip() else None
        if normalized is None and credential is None:
            raise ProviderConfigError(
                "AgentIdentity needs either an agent_id or a credential. A "
                "local identity must pass agent_id=... (its stable "
                "identifier); a verifiable identity derives the identifier "
                "from the IdP, so build it with a credential factory "
                "(AgentIdentity.from_entra/from_aws/from_gcp/from_spiffe/"
                "from_oidc)."
            )
        self.agent_id: str | None = normalized
        self.name: str | None = name
        self.owner: str | None = owner
        self.labels: dict[str, str] = dict(labels or {})
        self._credential: IdentityProvider | None = credential

    # -- Identity surface ------------------------------------------------

    @property
    def is_verifiable(self) -> bool:
        """``True`` when a credential provider backs this identity."""
        return self._credential is not None

    @property
    def credential_provider(self) -> str | None:
        """The backing provider's label (e.g. ``"entra-imds"``), if any."""
        return self._credential.provider_name if self._credential else None

    @property
    def credential(self) -> IdentityProvider | None:
        """The backing :class:`IdentityProvider` (advanced use)."""
        return self._credential

    def subject(self) -> str:
        """Return the IdP-assigned identifier from the credential.

        The persistent identity lives in the IdP; this reads who the IdP
        says the agent is — the ``sub`` claim (or ``oid`` for Microsoft
        Entra) of the verifiable credential. Acquiring the credential
        may contact the platform; the result is cached on the provider.

        .. note::
            The claim is read **without verifying the credential's
            signature** — the holder trusts its own IdP-issued token, and
            the receiving resource is what verifies the signature on
            presentation (see :class:`~promptise.mcp.server.JwksAuth`).
            For local attribution this is authoritative only insofar as
            the credential *source* is trusted; if you wire an untrusted
            ``token_fn`` / env var / file, the returned subject is only as
            trustworthy as that source.

        Returns:
            The IdP subject (``sub``) or, if absent, the object id
            (``oid``).

        Raises:
            ProviderConfigError: If this is a local identity with no
                credential backing.
            CredentialAcquisitionError: If the credential carries neither
                a ``sub`` nor an ``oid`` claim.
        """
        claims = self._require_idp_claims()
        for key in ("sub", "oid"):
            value = claims.get(key)
            if isinstance(value, str) and value:
                return value
        raise CredentialAcquisitionError(
            "The verifiable credential carries neither a 'sub' nor an 'oid' "
            "claim, so the IdP-assigned identity cannot be read. Most common "
            "cause: the credential is not an identity token (it has no "
            "subject)."
        )

    def idp_claims(self) -> dict[str, Any]:
        """Return the identity-relevant claims from the credential.

        A subset of the credential's claims that describe *who* the IdP
        says the agent is and *who issued* the identity: ``sub``, ``oid``,
        ``iss``, ``aud``, ``azp``, ``appid``.

        Raises:
            ProviderConfigError: If this is a local identity.
        """
        claims = self._require_idp_claims()
        return {k: claims[k] for k in ("sub", "oid", "iss", "aud", "azp", "appid") if k in claims}

    def resolve_identifier(self) -> str:
        """Return the authoritative identifier for this identity.

        The explicit ``agent_id`` handle when one was given, otherwise the
        IdP subject (which contacts the platform). This is the value used
        to attribute the agent's actions.
        """
        if self.agent_id is not None:
            return self.agent_id
        return self.subject()

    def _require_idp_claims(self) -> dict[str, Any]:
        if self._credential is None:
            raise ProviderConfigError(
                f"AgentIdentity {self.agent_id!r} is a local identity with no "
                f"IdP backing, so it has no IdP-assigned subject. Build it with "
                f"a credential factory to read the IdP identity."
            )
        return decode_jwt_claims(self._credential.get_credential())

    def claims(self) -> dict[str, Any]:
        """Return the identity as a dict for stamping onto traces/audit.

        Contains the ``agent_id`` and whichever of ``name``, ``owner``,
        ``credential_provider``, and ``labels`` are set, plus a
        ``verifiable`` flag. Never includes a credential token.
        """
        out: dict[str, Any] = {"verifiable": self.is_verifiable}
        if self.agent_id:
            out["agent_id"] = self.agent_id
        if self.name:
            out["name"] = self.name
        if self.owner:
            out["owner"] = self.owner
        if self.credential_provider:
            out["credential_provider"] = self.credential_provider
        if self.labels:
            out["labels"] = dict(self.labels)
        return out

    def get_credential(self, audience: str | None = None, *, force_refresh: bool = False) -> str:
        """Return a signed credential (JWT) proving this identity.

        Present this to a resource the agent calls — an MCP server or an
        HTTP API — so it can verify and attribute the caller.

        Args:
            audience: The resource the credential targets. When given, an
                IdP-backed identity whose provider can re-mint (Entra IMDS,
                AWS STS, GCP metadata, SPIFFE SDK) issues a credential
                scoped to that audience — so **one identity can present to
                several resources**. Projected-token and OIDC file/env
                modes have a fixed audience and ignore this. ``None`` uses
                the identity's configured default (and shares its cached
                credential).
            force_refresh: Skip the cached credential and acquire a fresh
                one, for when a resource rejected the cached credential
                before its ``exp`` (rotated or revoked key, clock skew).
                ``build_agent`` does this once when an MCP server answers
                ``401``.

        Returns:
            The credential JWT.

        Raises:
            ProviderConfigError: When this is a *local* identity with no
                verifiable credential backing.
            CredentialAcquisitionError: When the backing provider cannot
                mint a credential.
        """
        if self._credential is None:
            raise ProviderConfigError(
                f"AgentIdentity {self.agent_id!r} is a local identity with no "
                f"verifiable credential backing, so it cannot produce a "
                f"signed credential to present to a resource. Build it with a "
                f"credential factory — AgentIdentity.from_entra/from_aws/"
                f"from_gcp/from_spiffe/from_oidc — to make it verifiable."
            )
        return self._credential.get_credential(audience, force_refresh=force_refresh)

    def auth_header(self, audience: str | None = None) -> dict[str, str]:
        """Return ``{"Authorization": "Bearer <credential>"}``.

        Convenience for presenting the identity to an MCP server or API.
        Requires a verifiable identity (see :meth:`get_credential`).

        Args:
            audience: The resource the credential targets (see
                :meth:`get_credential`).
        """
        return {"Authorization": f"Bearer {self.get_credential(audience)}"}

    def __repr__(self) -> str:
        """Return an identifier-only repr — never includes a credential."""
        owner = f", owner={self.owner!r}" if self.owner else ""
        return (
            f"AgentIdentity(agent_id={self.agent_id!r}, name={self.name!r}"
            f"{owner}, verifiable={self.is_verifiable})"
        )

    # -- Verifiable-identity factories -----------------------------------

    @classmethod
    def from_entra(
        cls,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        mode: Literal["auto", "imds", "projected"] = "auto",
        client_id: str | None = None,
        token_file: str | Path | None = None,
        resource: str = _ENTRA_DEFAULT_RESOURCE,
    ) -> AgentIdentity:
        """Build an identity backed by Microsoft Entra ID.

        Args:
            agent_id: Stable identifier for the agent.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.
            mode: ``"auto"`` picks projected when
                ``$AZURE_FEDERATED_TOKEN_FILE`` is set, else IMDS.
            client_id: Managed-identity client id (IMDS mode).
            token_file: Projected token path (projected mode).
            resource: Resource/audience the credential targets.
        """
        return cls(
            agent_id,
            name=name,
            owner=owner,
            labels=labels,
            credential=_from_entra(
                mode=mode,
                client_id=client_id,
                token_file=token_file,
                resource=resource,
            ),
        )

    @classmethod
    def from_aws(
        cls,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        mode: Literal["auto", "sts", "projected"] = "auto",
        region: str | None = None,
        token_file: str | Path | None = None,
        audience: str = _AWS_DEFAULT_AUDIENCE,
        signing_algorithm: str = _AWS_DEFAULT_SIGNING_ALGORITHM,
    ) -> AgentIdentity:
        """Build an identity backed by AWS IAM.

        Args:
            agent_id: Stable identifier for the agent.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.
            mode: ``"auto"`` picks EKS-projected when
                ``$PROMPTISE_IDENTITY_TOKEN_FILE`` is set, else STS (on EKS
                with IRSA, STS uses the pod's role; IRSA's
                ``$AWS_WEB_IDENTITY_TOKEN_FILE`` is never presented).
            region: AWS region for STS. Falls back to ``$AWS_REGION``.
            token_file: EKS-projected token path.
            audience: Audience the credential targets.
            signing_algorithm: STS signing algorithm (e.g. ``"RS256"``).
        """
        return cls(
            agent_id,
            name=name,
            owner=owner,
            labels=labels,
            credential=_from_aws(
                mode=mode,
                region=region,
                token_file=token_file,
                audience=audience,
                signing_algorithm=signing_algorithm,
            ),
        )

    @classmethod
    def from_gcp(
        cls,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        audience: str = _GCP_DEFAULT_AUDIENCE,
        service_account_email: str = _GCP_DEFAULT_SERVICE_ACCOUNT_EMAIL,
        request_timeout: float = 5.0,
    ) -> AgentIdentity:
        """Build an identity backed by Google Cloud.

        Args:
            agent_id: Stable identifier for the agent.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.
            audience: Audience the credential targets.
            service_account_email: Attached service account whose
                identity to request (``"default"`` for the primary).
            request_timeout: Seconds to wait for the metadata response.
        """
        return cls(
            agent_id,
            name=name,
            owner=owner,
            labels=labels,
            credential=_from_gcp(
                audience=audience,
                service_account_email=service_account_email,
                request_timeout=request_timeout,
            ),
        )

    @classmethod
    def from_spiffe(
        cls,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        mode: Literal["auto", "file", "sdk"] = "auto",
        token_file: str | Path | None = None,
        socket_path: str | None = None,
        audience: str = _SPIFFE_DEFAULT_AUDIENCE,
    ) -> AgentIdentity:
        """Build an identity backed by SPIFFE / SPIRE.

        Args:
            agent_id: Stable identifier for the agent.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.
            mode: ``"auto"`` picks file mode when ``token_file`` is
                given, else SDK (Workload API) mode.
            token_file: JWT-SVID path (file mode).
            socket_path: Workload API socket (SDK mode). Falls back to
                ``$SPIFFE_ENDPOINT_SOCKET``.
            audience: Audience the JWT-SVID targets.
        """
        return cls(
            agent_id,
            name=name,
            owner=owner,
            labels=labels,
            credential=_from_spiffe(
                mode=mode,
                token_file=token_file,
                socket_path=socket_path,
                audience=audience,
            ),
        )

    @classmethod
    def from_oidc(
        cls,
        agent_id: str | None = None,
        *,
        issuer: str,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
        token_file: str | Path | None = None,
        token_fn: Callable[[], str] | None = None,
        token_env_var: str | None = None,
    ) -> AgentIdentity:
        """Build an identity backed by a generic OIDC issuer.

        Exactly one of ``token_file``, ``token_fn``, or ``token_env_var``
        must be supplied.

        Args:
            agent_id: Stable identifier for the agent.
            issuer: The OIDC issuer URL.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.
            token_file: Path to a file holding the JWT.
            token_fn: Zero-arg callable returning the JWT.
            token_env_var: Env var holding the JWT (re-read each refresh).
        """
        return cls(
            agent_id,
            name=name,
            owner=owner,
            labels=labels,
            credential=_from_oidc(
                issuer,
                token_file=token_file,
                token_fn=token_fn,
                token_env_var=token_env_var,
            ),
        )

    @classmethod
    def auto(
        cls,
        agent_id: str | None = None,
        *,
        name: str | None = None,
        owner: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> AgentIdentity:
        """Detect the platform and build a verifiable identity for it.

        Uses environment markers to pick Entra, AWS, GCP, or SPIFFE and
        dispatches to the matching factory with platform defaults. On AWS,
        ``$PROMPTISE_IDENTITY_TOKEN_FILE`` selects the projected-token mode
        and is itself an AWS marker; any other AWS marker (Lambda, ECS,
        ``EKS_POD_NAME``, IRSA's ``$AWS_WEB_IDENTITY_TOKEN_FILE``) selects
        STS mode.

        Args:
            agent_id: Stable identifier for the agent.
            name: Optional display name.
            owner: Optional owning team or person.
            labels: Optional free-form metadata.

        Raises:
            PlatformDetectionError: When no platform marker is present.
                Construct a local ``AgentIdentity(agent_id)`` or use an
                explicit factory instead.
        """
        platform = detect_platform()
        if platform == "entra":
            return cls.from_entra(agent_id, name=name, owner=owner, labels=labels)
        if platform == "aws":
            return cls.from_aws(agent_id, name=name, owner=owner, labels=labels)
        if platform == "gcp":
            return cls.from_gcp(agent_id, name=name, owner=owner, labels=labels)
        # The only remaining identifier from detect_platform() is "spiffe".
        return cls.from_spiffe(agent_id, name=name, owner=owner, labels=labels)
