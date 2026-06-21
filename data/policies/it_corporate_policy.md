# Enterprise IT and Security Policies

## Software Installation Policy (Third-Party)
All third-party software installations, including creative suites (e.g., Adobe Photoshop, Illustrator) and engineering tools (e.g., AutoCAD), require explicit authorization. The user must fill out the Software Request Form located on the internal intranet. Once signed by the department manager, IT Tier 2 will remotely deploy the package. Unauthorized installations will trigger a security alert.

## Password Management and Reset Protocol
Users experiencing expired passwords or lockout issues must utilize the Self-Service Password Reset (SSPR) portal at sspr.company.com. The portal requires the user's mobile device for Multi-Factor Authentication (MFA) via SMS or Authenticator app. IT Support is strictly prohibited from manually resetting passwords over email or chat without visual or voice verification.

## Production Server Incident Response (Code 500/503)
Any HTTP 500 (Internal Server Error) or HTTP 503 (Service Unavailable) errors occurring on production servers (IP range 192.168.1.x) are classified as SEV-1 (Critical). Tier 2 support must NOT attempt to reboot, restart services, or SSH into these machines. The correct protocol is to immediately file an Internal Escalation Log and page the DevOps Tier 3 on-call engineer via the PagerDuty emergency channel.

## Hardware Replacement Guidelines
If a user reports physical damage to corporate hardware (laptops, monitors), Tier 1 must request photographic evidence. If the device is unbootable, a loaner device will be issued within 24 hours. Data recovery is not guaranteed.