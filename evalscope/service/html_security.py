"""Security headers for interactive HTML reports served by the application."""

from __future__ import annotations

from flask import Response

from evalscope.constants import PLOTLY_CDN_URL


def report_content_security_policy(plotly_source: str = PLOTLY_CDN_URL) -> str:
    return '; '.join((
        "default-src 'none'",
        f"script-src 'unsafe-inline' {plotly_source}",
        "style-src 'unsafe-inline'",
        "img-src data: blob:",
        "font-src data:",
        "connect-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'self'",
        "sandbox allow-scripts",
    ))


REPORT_CONTENT_SECURITY_POLICY = report_content_security_policy()


def secure_report_response(response: Response, plotly_source: str = PLOTLY_CDN_URL) -> Response:
    """Apply a fail-closed browser sandbox to an HTML report response."""
    response.headers['Content-Security-Policy'] = report_content_security_policy(plotly_source)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response
