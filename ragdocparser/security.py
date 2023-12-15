"""
Advanced Security Module for RAGDocParser.
Enterprise-grade security features for document processing and data protection.
"""

import logging
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    encryption_enabled: bool = True
    access_logging: bool = True
    rate_limiting: bool = True
    audit_trail: bool = True
    pii_detection: bool = True
    content_filtering: bool = True
    jwt_secret: str = None
    session_timeout: int = 3600  # 1 hour
    max_requests_per_minute: int = 100
    
    def __post_init__(self):
        if self.jwt_secret is None:
            self.jwt_secret = secrets.token_urlsafe(32)

class ContentScanner:
    """Scanner for detecting sensitive content and PII."""
    
    def __init__(self):
        """Initialize content scanner."""
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        self.sensitive_keywords = [
            'confidential', 'secret', 'private', 'classified',
            'restricted', 'internal', 'proprietary', 'sensitive'
        ]
    
    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for sensitive information."""
        import re
        
        findings = {
            'pii_detected': {},
            'sensitive_keywords': [],
            'risk_level': 'low',
            'recommendations': []
        }
        
        # Scan for PII patterns
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                findings['pii_detected'][pii_type] = len(matches)
        
        # Scan for sensitive keywords
        content_lower = content.lower()
        for keyword in self.sensitive_keywords:
            if keyword in content_lower:
                findings['sensitive_keywords'].append(keyword)
        
        # Determine risk level
        pii_count = sum(findings['pii_detected'].values())
        keyword_count = len(findings['sensitive_keywords'])
        
        if pii_count > 5 or keyword_count > 3:
            findings['risk_level'] = 'high'
            findings['recommendations'].append('Consider encrypting this document')
            findings['recommendations'].append('Limit access to authorized personnel only')
        elif pii_count > 0 or keyword_count > 0:
            findings['risk_level'] = 'medium'
            findings['recommendations'].append('Review content before sharing')
        
        return findings
    
    def redact_content(self, content: str, redaction_char: str = '*') -> str:
        """Redact sensitive information from content."""
        import re
        
        redacted_content = content
        
        # Redact PII
        for pii_type, pattern in self.pii_patterns.items():
            def redact_match(match):
                return redaction_char * len(match.group())
            
            redacted_content = re.sub(pattern, redact_match, redacted_content, flags=re.IGNORECASE)
        
        return redacted_content

class EncryptionManager:
    """Manager for document encryption and decryption."""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize encryption manager."""
        self.password = password or os.environ.get('RAG_ENCRYPTION_PASSWORD')
        if not self.password:
            self.password = secrets.token_urlsafe(32)
            logger.warning("No encryption password provided, generated random password")
        
        self.fernet = self._create_fernet_cipher()
    
    def _create_fernet_cipher(self) -> Fernet:
        """Create Fernet cipher from password."""
        password_bytes = self.password.encode()
        salt = b'salt_1234567890123456'  # In production, use random salt per document
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt_content(self, content: str) -> str:
        """Encrypt content."""
        try:
            content_bytes = content.encode('utf-8')
            encrypted_bytes = self.fernet.encrypt(content_bytes)
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt content."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_content.encode('utf-8'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Encrypt a file."""
        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + '.encrypted')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        encrypted_content = self.encrypt_content(content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(encrypted_content)
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt a file."""
        if output_path is None:
            output_path = encrypted_file_path.with_suffix('')
        
        with open(encrypted_file_path, 'r', encoding='utf-8') as f:
            encrypted_content = f.read()
        
        decrypted_content = self.decrypt_content(encrypted_content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(decrypted_content)
        
        return output_path

class AccessControlManager:
    """Manager for access control and user authentication."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize access control manager."""
        self.config = config
        self.active_sessions = {}
        self.user_permissions = {}
        self.access_logs = []
        self._lock = threading.RLock()
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        # Simple authentication (in production, use proper user database)
        if self._verify_credentials(username, password):
            payload = {
                'username': username,
                'exp': datetime.utcnow() + timedelta(seconds=self.config.session_timeout),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
            
            with self._lock:
                self.active_sessions[token] = {
                    'username': username,
                    'login_time': datetime.utcnow(),
                    'last_activity': datetime.utcnow()
                }
            
            self._log_access('login', username, success=True)
            return token
        else:
            self._log_access('login', username, success=False)
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            
            with self._lock:
                if token in self.active_sessions:
                    self.active_sessions[token]['last_activity'] = datetime.utcnow()
                    return {
                        'username': payload['username'],
                        'session_info': self.active_sessions[token]
                    }
            
            return None
            
        except jwt.InvalidTokenError:
            return None
    
    def logout_user(self, token: str) -> bool:
        """Logout user and invalidate token."""
        with self._lock:
            if token in self.active_sessions:
                username = self.active_sessions[token]['username']
                del self.active_sessions[token]
                self._log_access('logout', username, success=True)
                return True
        
        return False
    
    def set_user_permissions(self, username: str, permissions: List[str]) -> None:
        """Set permissions for a user."""
        with self._lock:
            self.user_permissions[username] = permissions
    
    def check_permission(self, username: str, permission: str) -> bool:
        """Check if user has specific permission."""
        with self._lock:
            user_perms = self.user_permissions.get(username, [])
            return permission in user_perms or 'admin' in user_perms
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (simplified for demo)."""
        # In production, use proper password hashing and user database
        test_users = {
            'admin': 'admin_password',
            'user1': 'user1_password',
            'demo': 'demo_password'
        }
        
        return test_users.get(username) == password
    
    def _log_access(self, action: str, username: str, success: bool, details: str = '') -> None:
        """Log access attempt."""
        if self.config.access_logging:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': action,
                'username': username,
                'success': success,
                'details': details,
                'ip_address': 'unknown'  # Would be filled from request context
            }
            
            with self._lock:
                self.access_logs.append(log_entry)
                
                # Keep only last 1000 log entries
                if len(self.access_logs) > 1000:
                    self.access_logs = self.access_logs[-1000:]
    
    def get_access_logs(self, username: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access logs."""
        with self._lock:
            logs = self.access_logs
            
            if username:
                logs = [log for log in logs if log['username'] == username]
            
            return logs[-limit:]

class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_counts = {}
        self._lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        current_time = datetime.utcnow()
        
        with self._lock:
            if identifier not in self.request_counts:
                self.request_counts[identifier] = []
            
            # Clean old requests outside time window
            cutoff_time = current_time - timedelta(seconds=self.time_window)
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > cutoff_time
            ]
            
            # Check if under limit
            if len(self.request_counts[identifier]) < self.max_requests:
                self.request_counts[identifier].append(current_time)
                return True
            else:
                return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        current_time = datetime.utcnow()
        
        with self._lock:
            if identifier not in self.request_counts:
                return self.max_requests
            
            # Clean old requests
            cutoff_time = current_time - timedelta(seconds=self.time_window)
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > cutoff_time
            ]
            
            return max(0, self.max_requests - len(self.request_counts[identifier]))

class SecurityManager:
    """Main security manager for RAGDocParser."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security manager."""
        self.config = config or SecurityConfig()
        self.content_scanner = ContentScanner()
        self.encryption_manager = EncryptionManager() if self.config.encryption_enabled else None
        self.access_control = AccessControlManager(self.config)
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute) if self.config.rate_limiting else None
        self.audit_trail = []
    
    def scan_document_security(self, content: str, filename: str = 'unknown') -> Dict[str, Any]:
        """Comprehensive security scan of document content."""
        scan_results = {
            'filename': filename,
            'scan_timestamp': datetime.utcnow().isoformat(),
            'content_length': len(content),
            'security_findings': {}
        }
        
        if self.config.pii_detection:
            pii_results = self.content_scanner.scan_content(content)
            scan_results['security_findings']['pii_scan'] = pii_results
        
        if self.config.content_filtering:
            # Additional content filtering logic
            scan_results['security_findings']['content_safe'] = self._check_content_safety(content)
        
        # Log audit trail
        if self.config.audit_trail:
            self._add_audit_entry('document_scanned', {'filename': filename, 'findings': scan_results['security_findings']})
        
        return scan_results
    
    def secure_document_processing(self, content: str, user_token: Optional[str] = None) -> Dict[str, Any]:
        """Process document with security checks."""
        result = {
            'success': False,
            'message': '',
            'processed_content': None,
            'security_info': {}
        }
        
        # Verify user authentication
        if user_token:
            user_info = self.access_control.verify_token(user_token)
            if not user_info:
                result['message'] = 'Invalid or expired token'
                return result
        
        # Rate limiting check
        if self.rate_limiter and user_token:
            user_id = user_info['username'] if user_info else 'anonymous'
            if not self.rate_limiter.is_allowed(user_id):
                result['message'] = 'Rate limit exceeded'
                return result
        
        # Security scan
        security_scan = self.scan_document_security(content)
        result['security_info'] = security_scan
        
        # Check if content is safe to process
        if security_scan['security_findings'].get('pii_scan', {}).get('risk_level') == 'high':
            result['message'] = 'Document contains high-risk sensitive information'
            return result
        
        # Process content (redact if needed)
        if self.config.pii_detection and security_scan['security_findings'].get('pii_scan', {}).get('pii_detected'):
            processed_content = self.content_scanner.redact_content(content)
        else:
            processed_content = content
        
        # Encrypt if enabled
        if self.encryption_manager:
            processed_content = self.encryption_manager.encrypt_content(processed_content)
        
        result['success'] = True
        result['message'] = 'Document processed successfully'
        result['processed_content'] = processed_content
        
        return result
    
    def _check_content_safety(self, content: str) -> bool:
        """Check if content is safe for processing."""
        # Simple content safety check (expand as needed)
        unsafe_patterns = ['malware', 'virus', 'exploit', 'hack']
        content_lower = content.lower()
        
        for pattern in unsafe_patterns:
            if pattern in content_lower:
                return False
        
        return True
    
    def _add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        if self.config.audit_trail:
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': action,
                'details': details
            }
            
            self.audit_trail.append(audit_entry)
            
            # Keep only last 1000 audit entries
            if len(self.audit_trail) > 1000:
                self.audit_trail = self.audit_trail[-1000:]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        return {
            'config': {
                'encryption_enabled': self.config.encryption_enabled,
                'access_logging': self.config.access_logging,
                'rate_limiting': self.config.rate_limiting,
                'pii_detection': self.config.pii_detection
            },
            'active_sessions': len(self.access_control.active_sessions),
            'audit_entries': len(self.audit_trail),
            'recent_access_logs': self.access_control.get_access_logs(limit=10)
        }
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired_count = 0
        current_time = datetime.utcnow()
        
        with self.access_control._lock:
            expired_tokens = []
            for token, session_info in self.access_control.active_sessions.items():
                session_age = current_time - session_info['last_activity']
                if session_age.total_seconds() > self.config.session_timeout:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.access_control.active_sessions[token]
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count

# Global security manager
_global_security_manager: Optional[SecurityManager] = None

def get_security_manager() -> SecurityManager:
    """Get or create the global security manager."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager

def scan_for_pii(content: str) -> Dict[str, Any]:
    """Convenience function to scan content for PII."""
    manager = get_security_manager()
    return manager.content_scanner.scan_content(content)

def authenticate_user(username: str, password: str) -> Optional[str]:
    """Convenience function for user authentication."""
    manager = get_security_manager()
    return manager.access_control.authenticate_user(username, password) 