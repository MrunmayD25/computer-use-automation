"""Independent fixture checks, never passed to discovery or production code."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import Page, Request, Response

from computeruse.capability import Field, ValueType
from computeruse.contract import Binding, Contract
from computeruse.recording import field

STATUSES = ("active", "inactive", "deceased", "merged")

PRODUCTS = tuple(
    f"{currency} {kind}"
    for currency in ("EUR", "JPY", "KWD", "USD")
    for kind in ("checking", "savings")
)
"""The open deposit products every evaluation site offers, by their shown name."""


OPERATOR = (Binding("operator_id", "Operator"),)
"""The sign-on form's operator field, which a site may open already set."""


CONTRACT = Contract(
    2,
    (field("member_id"), field("operator_id")),
    (
        Field(
            "membership_status",
            ValueType.CHOICE,
            True,
            1,
            20,
            STATUSES + tuple(value.title() for value in STATUSES),
        ),
    ),
    OPERATOR,
)


OPEN_ACCOUNT = Contract(
    2,
    (
        field("member_id"),
        field("operator_id"),
        Field("product", ValueType.CHOICE, True, 1, 40, PRODUCTS),
        field("nickname"),
        Field(
            "statement_delivery",
            ValueType.CHOICE,
            True,
            1,
            20,
            ("paper", "electronic"),
        ),
    ),
    (Field("account_number", ValueType.TEXT, True, 1, 40, ()),),
    OPERATOR,
)
"""The member, operator, product, and new number for an opened account."""


def canonical_status(value: str | None) -> str | None:
    """Map the fixture's display casing to its closed status enum."""
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized if normalized in STATUSES else None


def tokens(text: str) -> list[str]:
    """Split context labels at literal delimiters without loose substrings."""
    for separator in "()[]:|,·":
        text = text.replace(separator, " ")
    return text.split()


class ContextOracle:
    """Check actual fixture context separately from the model's completion claim."""

    def __init__(self, page: Page, app: str, operator: str) -> None:
        self.page = page
        self.app = app
        self.operator = operator
        self.signed_on = False
        self.origin = urlsplit(page.url).netloc
        # The two web portals bind Northstar as the first sign-on option.
        # Track the accepted login flow without passing this fact to the model.
        page.context.on("request", self._request)
        page.context.on("response", self._response)

    def _request(self, request: Request) -> None:
        url = urlsplit(request.url)
        if url.netloc == self.origin and url.path in {"/signout", "/signon"}:
            self.signed_on = False

    def _response(self, response: Response) -> None:
        request = response.request
        url = urlsplit(request.url)
        if (
            url.netloc == self.origin
            and request.method == "POST"
            and url.path == "/signon"
        ):
            fields = parse_qs(request.post_data or "")
            self.signed_on = (
                response.status == 303
                and response.headers.get("location") == "/"
                and fields.get("binding") == ["0"]
                and fields.get("actorNumber") == [self.operator]
            )

    def check(self) -> dict[str, bool]:
        """Return comparison booleans only. Unverifiable context fails the check."""
        page = self.page
        if self.app.startswith("white-label-"):
            operator = " ".join(
                page.locator(".operator, .context-id").all_text_contents()
            )
            return {
                "operator": self.operator in tokens(operator),
                "institution": self.signed_on,
            }
        if self.app == "web-component-operations":
            operator = " ".join(page.locator("ops-app .operator").all_text_contents())
            institution = " ".join(page.locator("ops-app .inst").all_text_contents())
        elif self.app == "canvas-teller":
            # The canvas paints its header, so the oracle asks the server for
            # the session the page's own cookie holds.
            session = page.request.get(f"http://{self.origin}/api/session").json()
            signed = session.get("operator") or {}
            operator = str(signed.get("number") or "")
            institution = str(session.get("institution") or "")
        else:
            return {"operator": False, "institution": False}
        return {
            "operator": self.operator in tokens(operator),
            "institution": "northstar" in tokens(institution.casefold()),
        }
