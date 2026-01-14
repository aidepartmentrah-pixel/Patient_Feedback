"""
Compliance Evaluator Service
Evaluates seasonal report compliance against organizational unit policies.
"""

from typing import Dict, Any, List, Optional


class ComplianceEvaluator:
    """
    Evaluates compliance by comparing aggregated seasonal data against policy thresholds.
    
    Compliance Rules:
    - Target 1: Domain distribution percentage limits
    - Target 2: High severity case limits (absolute or percentage-based)
    """
    
    def evaluate(
        self,
        totals: Dict[str, int],
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate compliance against policy thresholds.
        
        Args:
            totals: Aggregated seasonal totals:
                {
                    'total_cases': int,
                    'clinical_domain_count': int,
                    'management_domain_count': int,
                    'relational_domain_count': int,
                    'low_severity_count': int,
                    'medium_severity_count': int,
                    'high_severity_count': int
                }
            
            policy: Policy configuration:
                {
                    'MaxAllowedCases': int,
                    'MaxClinicalDomain': int,
                    'MaxManagementDomain': int,
                    'MaxRelationalDomain': int,
                    'RequireExplanationAboveThreshold': bool,
                    'EscalationEnabled': bool,
                    'EscalationThresholdPercentage': int
                }
        
        Returns:
            Dictionary with compliance result:
            {
                'is_compliant': bool,
                'violated_rules': str | None,
                'explanation_status_id': int
            }
        """
        violations = []
        
        total_cases = totals.get('total_cases', 0)
        
        # Guard against division by zero
        if total_cases == 0:
            # No cases = compliant by default
            return {
                'is_compliant': True,
                'violated_rules': None,
                'explanation_status_id': 4  # No Explanation Needed
            }
        
        # -----------------------------
        # Target 1: Domain Distribution Limits
        # -----------------------------
        domain_violations = self._evaluate_domain_limits(totals, policy, total_cases)
        violations.extend(domain_violations)
        
        # -----------------------------
        # Target 2: High Severity Limits
        # -----------------------------
        severity_violations = self._evaluate_high_severity_limits(totals, policy, total_cases)
        violations.extend(severity_violations)
        
        # -----------------------------
        # Build Compliance Result
        # -----------------------------
        is_compliant = len(violations) == 0
        violated_rules = None if is_compliant else " | ".join(violations)
        
        # Determine explanation status
        if is_compliant:
            explanation_status_id = 4  # No Explanation Needed
        else:
            explanation_status_id = 1  # Waiting (explanation required)
        
        return {
            'is_compliant': is_compliant,
            'violated_rules': violated_rules,
            'explanation_status_id': explanation_status_id
        }
    
    def _evaluate_domain_limits(
        self,
        totals: Dict[str, int],
        policy: Dict[str, Any],
        total_cases: int
    ) -> List[str]:
        """
        Evaluate domain distribution percentage limits.
        
        Args:
            totals: Aggregated totals
            policy: Policy configuration
            total_cases: Total case count (for percentage calculation)
        
        Returns:
            List of violation messages
        """
        violations = []
        
        # Domain mappings: (count_key, policy_limit_key, display_name)
        domains = [
            ('clinical_domain_count', 'MaxClinicalDomain', 'Clinical domain'),
            ('management_domain_count', 'MaxManagementDomain', 'Management domain'),
            ('relational_domain_count', 'MaxRelationalDomain', 'Relational domain')
        ]
        
        for count_key, policy_key, display_name in domains:
            domain_count = totals.get(count_key, 0)
            policy_limit = policy.get(policy_key)
            
            # Skip if no policy limit defined
            if policy_limit is None:
                continue
            
            # Calculate actual percentage
            domain_percentage = (domain_count / total_cases) * 100.0
            
            # Check violation
            if domain_percentage > policy_limit:
                violations.append(
                    f"{display_name} exceeded limit (actual {domain_percentage:.1f}%, limit {policy_limit}%)"
                )
        
        return violations
    
    def _evaluate_high_severity_limits(
        self,
        totals: Dict[str, int],
        policy: Dict[str, Any],
        total_cases: int
    ) -> List[str]:
        """
        Evaluate high severity case limits.
        Supports both absolute count and percentage-based limits.
        
        Args:
            totals: Aggregated totals
            policy: Policy configuration
            total_cases: Total case count (for percentage calculation)
        
        Returns:
            List of violation messages
        """
        violations = []
        
        high_severity_count = totals.get('high_severity_count', 0)
        
        # Check if high severity limit is defined
        # NOTE: Policy field name assumed based on context - may need adjustment
        high_severity_limit = policy.get('HighSeverityLimit') or policy.get('MaxHighSeverity')
        
        if high_severity_limit is None:
            # No high severity policy defined
            return violations
        
        # Check if percentage-based rule is enabled
        enable_percentage_rule = policy.get('EnableHighSeverityPercentageRule', False)
        
        if enable_percentage_rule:
            # Percentage-based evaluation
            high_severity_percentage = (high_severity_count / total_cases) * 100.0
            
            if high_severity_percentage > high_severity_limit:
                violations.append(
                    f"High severity cases exceeded percentage limit "
                    f"(actual {high_severity_percentage:.1f}%, limit {high_severity_limit}%)"
                )
        else:
            # Absolute count evaluation
            if high_severity_count > high_severity_limit:
                violations.append(
                    f"High severity cases exceeded absolute limit "
                    f"(actual {high_severity_count}, limit {high_severity_limit})"
                )
        
        return violations
