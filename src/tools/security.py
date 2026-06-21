import re

class SecurityGuard:
    def __init__(self):
        # We define regular expressions to catch sensitive data
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
        
    def sanitize_output(self, text: str) -> dict:
        """
        Scans the agent's response, masks sensitive information, 
        and returns an audit log of what was redacted.
        """
        audit_log = []
        sanitized_text = text
        
        # 1. Detect and Mask IP addresses
        ips_found = self.ip_pattern.findall(sanitized_text)
        if ips_found:
            audit_log.append(f"Blocked {len(ips_found)} Internal IP Address(es)")
            sanitized_text = self.ip_pattern.sub("[REDACTED INTERNAL IP]", sanitized_text)
            
        # 2. Detect and Mask Credit Cards
        ccs_found = self.cc_pattern.findall(sanitized_text)
        if ccs_found:
            audit_log.append(f"Blocked {len(ccs_found)} Credit Card Number(s)")
            sanitized_text = self.cc_pattern.sub("[REDACTED CREDIT CARD]", sanitized_text)
            
        return {
            "safe_text": sanitized_text,
            "interventions": audit_log
        }