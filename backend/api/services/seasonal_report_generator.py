"""
Seasonal Report Generator Service
Orchestrates the generation of materialized seasonal reports.
"""

from typing import Dict, Any
from backend.api.db_layer.seasonal_report import (
    get_existing_seasonal_report_id,
    update_seasonal_report_header,
    insert_seasonal_report_header,
    delete_seasonal_report_classification_stats,
    delete_seasonal_report_policy_snapshot,
    insert_seasonal_report_classification_stats,
    insert_seasonal_report_policy_snapshot
)


class SeasonalReportGenerator:
    """
    Orchestrates seasonal report generation process.
    Coordinates aggregation, compliance evaluation, and persistence.
    """
    
    def generate_or_regenerate_report(
        self,
        season_id: int,
        orgunit_id: int,
        orgunit_type: int,
        generated_by_user_id: int
    ) -> Dict[str, Any]:
        """
        Generate or regenerate a seasonal report for an organizational unit.
        
        Regeneration Strategy:
        1. Delete existing report (CASCADE removes children + action items)
        2. Aggregate fresh data from raw cases
        3. Evaluate compliance against policy
        4. Insert new report header
        5. Insert classification statistics
        6. Insert policy snapshot
        
        Args:
            season_id: Season identifier
            orgunit_id: Organizational unit identifier
            orgunit_type: Type of organizational unit (1=Dept, 2=Building, 3=Org)
            generated_by_user_id: User who triggered the generation
        
        Returns:
            Dictionary with report summary:
            {
                'seasonal_report_id': int,
                'season_id': int,
                'orgunit_id': int,
                'orgunit_type': int,
                'status': 'generated'
            }
        """
        
        # -----------------------------
        # STEP 1: Check for Existing Report
        # -----------------------------
        # Check if report already exists - if so, UPDATE instead of DELETE+INSERT
        # This preserves the SeasonalReportID and linked action items
        existing_report_id = get_existing_seasonal_report_id(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
        
        # -----------------------------
        # STEP 2: Aggregate Raw Data
        # -----------------------------
        print(f"[GENERATOR] Starting aggregation for season_id={season_id}, orgunit_id={orgunit_id}, orgunit_type={orgunit_type}")
        
        # Fetch classification-level statistics from incident cases
        print("[GENERATOR] Fetching classification stats...")
        classification_stats = self._aggregate_classification_stats(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
        print(f"[GENERATOR] Got {len(classification_stats)} classification stats")
        
        # Fetch domain and severity totals
        print("[GENERATOR] Fetching domain totals...")
        domain_totals = self._aggregate_domain_totals(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
        print(f"[GENERATOR] Domain totals: {domain_totals}")
        
        # Fetch current policy configuration for snapshot
        print("[GENERATOR] Fetching policy snapshot...")
        policy_snapshot = self._get_policy_snapshot(
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
        print(f"[GENERATOR] Policy snapshot: {policy_snapshot}")
        
        # -----------------------------
        # STEP 3: Evaluate Compliance
        # -----------------------------
        print("[GENERATOR] Evaluating compliance...")
        compliance_result = self._evaluate_compliance(
            domain_totals=domain_totals,
            policy_snapshot=policy_snapshot
        )
        print(f"[GENERATOR] Compliance result: {compliance_result}")
        
        # -----------------------------
        # STEP 4: Update or Insert Report Header
        # -----------------------------
        if existing_report_id:
            # UPDATE existing report (preserves action items)
            update_seasonal_report_header(
                seasonal_report_id=existing_report_id,
                total_cases=domain_totals['total_cases'],
                low_severity_count=domain_totals['low_severity_count'],
                medium_severity_count=domain_totals['medium_severity_count'],
                high_severity_count=domain_totals['high_severity_count'],
                clinical_domain_count=domain_totals['clinical_domain_count'],
                management_domain_count=domain_totals['management_domain_count'],
                relational_domain_count=domain_totals['relational_domain_count'],
                is_compliant=compliance_result['is_compliant'],
                violated_rules=compliance_result['violated_rules'],
                explanation_status_id=compliance_result['explanation_status_id'],
                updated_by_user_id=generated_by_user_id
            )
            seasonal_report_id = existing_report_id
        else:
            # INSERT new report
            seasonal_report_id = insert_seasonal_report_header(
                season_id=season_id,
                orgunit_id=orgunit_id,
                orgunit_type=orgunit_type,
                total_cases=domain_totals['total_cases'],
                low_severity_count=domain_totals['low_severity_count'],
                medium_severity_count=domain_totals['medium_severity_count'],
                high_severity_count=domain_totals['high_severity_count'],
                clinical_domain_count=domain_totals['clinical_domain_count'],
                management_domain_count=domain_totals['management_domain_count'],
                relational_domain_count=domain_totals['relational_domain_count'],
                is_compliant=compliance_result['is_compliant'],
                violated_rules=compliance_result['violated_rules'],
                explanation_status_id=compliance_result['explanation_status_id'],
                created_by_user_id=generated_by_user_id
            )
        
        # -----------------------------
        # STEP 5: Refresh Classification Stats
        # -----------------------------
        # Delete old stats and insert fresh ones
        if existing_report_id:
            delete_seasonal_report_classification_stats(seasonal_report_id)
        
        if classification_stats:
            insert_seasonal_report_classification_stats(
                seasonal_report_id=seasonal_report_id,
                stats=classification_stats
            )
        
        # -----------------------------
        # STEP 6: Refresh Policy Snapshot
        # -----------------------------
        # Delete old snapshot and insert fresh one
        if existing_report_id:
            delete_seasonal_report_policy_snapshot(seasonal_report_id)
        
        if policy_snapshot:
            insert_seasonal_report_policy_snapshot(
                seasonal_report_id=seasonal_report_id,
                policy_row=policy_snapshot
            )
        
        # -----------------------------
        # STEP 6.5: API V2 ADAPTER HOOK (SAFE / NON-BLOCKING)
        # Automatically create subcases for this seasonal report
        #
        # Skipped when the unit has zero cases this season: a 0-case report
        # is always is_compliant=True (every rule in _evaluate_compliance
        # either multiplies by total_cases or is gated on total_cases > 0),
        # so there is never anything for the unit to explain — creating a
        # subcase here would only produce a no-op inbox notification telling
        # them they have 0 cases. If real cases appear later in the same
        # season, the next regeneration will see total_cases > 0 and create
        # the subcase then (create_subcases_for_seasonal_report is itself
        # idempotent per report, so this can't double-create).
        # -----------------------------
        if domain_totals['total_cases'] > 0:
            try:
                from backend.api_v2.services.case_creation_service import create_subcases_for_seasonal_report
                # Note: current_user is not available in this legacy code context
                # We'll pass None and the service will handle it gracefully
                create_subcases_for_seasonal_report(seasonal_report_id, current_user=None)
            except Exception as e:
                # Log only — never interrupt main flow
                print(f"[API V2 ADAPTER WARNING] Failed to create subcases for seasonal report {seasonal_report_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # -----------------------------
        # STEP 7: Return Summary
        # -----------------------------
        return {
            'seasonal_report_id': seasonal_report_id,
            'season_id': season_id,
            'orgunit_id': orgunit_id,
            'orgunit_type': orgunit_type,
            'status': 'generated',
            'total_cases': domain_totals['total_cases'],
            'is_compliant': compliance_result['is_compliant']
        }
    
    # -----------------------------
    # Private Aggregation Methods (PLACEHOLDERS)
    # -----------------------------
    
    def _aggregate_classification_stats(
        self,
        season_id: int,
        orgunit_id: int,
        orgunit_type: int
    ) -> list:
        """
        Aggregate per-classification statistics from raw incident cases.
        
        Args:
            season_id: Season identifier
            orgunit_id: Organizational unit identifier
            orgunit_type: Type of organizational unit
        
        Returns:
            List of classification statistics dictionaries:
            [
                {
                    'classification_id': int,
                    'total_count': int,
                    'low_count': int,
                    'medium_count': int,
                    'high_count': int,
                    'preventive_yes_count': int,
                    'preventive_no_count': int
                },
                ...
            ]
        """
        from backend.api.db_layer.seasonal_report_aggregation import get_seasonal_classification_stats
        return get_seasonal_classification_stats(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
    
    def _aggregate_domain_totals(
        self,
        season_id: int,
        orgunit_id: int,
        orgunit_type: int
    ) -> dict:
        """
        Aggregate domain and severity totals from raw incident cases.
        
        Args:
            season_id: Season identifier
            orgunit_id: Organizational unit identifier
            orgunit_type: Type of organizational unit
        
        Returns:
            Dictionary with domain and severity totals:
            {
                'total_cases': int,
                'clinical_domain_count': int,
                'management_domain_count': int,
                'relational_domain_count': int,
                'low_severity_count': int,
                'medium_severity_count': int,
                'high_severity_count': int
            }
        """
        from backend.api.db_layer.seasonal_report_aggregation import get_seasonal_domain_totals
        return get_seasonal_domain_totals(
            season_id=season_id,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
    
    def _get_policy_snapshot(
        self,
        orgunit_id: int,
        orgunit_type: int
    ) -> dict:
        """
        Fetch current policy configuration for snapshot.
        
        Args:
            orgunit_id: Organizational unit identifier
            orgunit_type: Type of organizational unit
        
        Returns:
            Dictionary with policy fields from APP_OrgUnitPolicy:
            {
                'OrgUnitID': int,
                'OrgUnitType': int,
                'MaxAllowedCases': int,
                'MaxClinicalDomain': int,
                'MaxManagementDomain': int,
                'MaxRelationalDomain': int,
                'RequireExplanationAboveThreshold': bool,
                'EscalationEnabled': bool,
                'EscalationThresholdPercentage': int
            }
        """
        from backend.api.db_layer.seasonal_report_aggregation import get_orgunit_policy_row
        return get_orgunit_policy_row(
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type
        )
    
    def _evaluate_compliance(
        self,
        domain_totals: dict,
        policy_snapshot: dict
    ) -> dict:
        """
        Evaluate compliance by comparing aggregated data against policy thresholds.
        Only evaluates rules that are explicitly enabled via Enable* flags.
        
        Domain limits are evaluated as PERCENTAGES.
        
        Args:
            domain_totals: Aggregated domain and severity totals
            policy_snapshot: Current policy configuration with Enable flags
        
        Returns:
            Dictionary with compliance evaluation:
            {
                'is_compliant': bool,
                'violated_rules': str | None (JSON string),
                'explanation_status_id': int
            }
        """
        import json
        
        # If no policy exists, default to compliant
        if not policy_snapshot:
            return {
                'is_compliant': True,
                'violated_rules': None,
                'explanation_status_id': 1  # 1 = Not Required
            }
        
        violations = []
        total_cases = domain_totals.get('total_cases', 0)
        
        # Get enable flags (convert to bool to handle 0/1 from database)
        enable_domain_rule = bool(policy_snapshot.get('EnableHighSeverityPercentageByDomainRule', False))
        enable_low_rule = bool(policy_snapshot.get('EnableLowSeverityRepetitionRule', False))
        enable_medium_rule = bool(policy_snapshot.get('EnableMediumSeverityRepetitionRule', False))
        enable_high_rule = bool(policy_snapshot.get('EnableHighSeverityPercentageRule', False))
        
        print(f"[EVALUATOR] Enable flags: domain={enable_domain_rule}, low={enable_low_rule}, med={enable_medium_rule}, high={enable_high_rule}")
        
        # ============================================================
        # Check domain-specific thresholds (PERCENTAGES)
        # Only if EnableHighSeverityPercentageByDomainRule is enabled
        # ============================================================
        if enable_domain_rule and total_cases > 0:
            # Clinical Domain
            clinical_limit = policy_snapshot.get('ClinicalDomainLimit', 0)
            clinical_count = domain_totals.get('clinical_domain_count', 0)
            clinical_percentage = (clinical_count / total_cases) * 100
            
            if clinical_limit > 0 and clinical_percentage > clinical_limit:
                violations.append({
                    'rule': 'ClinicalDomainLimit',
                    'rule_name_ar': 'المجال السريري',
                    'rule_name_en': 'Clinical Domain',
                    'threshold': clinical_limit,
                    'threshold_unit': '%',
                    'actual': round(clinical_percentage, 1),
                    'actual_unit': '%',
                    'enabled': True
                })
            
            # Management Domain
            management_limit = policy_snapshot.get('ManagementDomainLimit', 0)
            management_count = domain_totals.get('management_domain_count', 0)
            management_percentage = (management_count / total_cases) * 100
            
            if management_limit > 0 and management_percentage > management_limit:
                violations.append({
                    'rule': 'ManagementDomainLimit',
                    'rule_name_ar': 'المجال الإداري',
                    'rule_name_en': 'Management Domain',
                    'threshold': management_limit,
                    'threshold_unit': '%',
                    'actual': round(management_percentage, 1),
                    'actual_unit': '%',
                    'enabled': True
                })
            
            # Relational Domain
            relational_limit = policy_snapshot.get('RelationalDomainLimit', 0)
            relational_count = domain_totals.get('relational_domain_count', 0)
            relational_percentage = (relational_count / total_cases) * 100
            
            if relational_limit > 0 and relational_percentage > relational_limit:
                violations.append({
                    'rule': 'RelationalDomainLimit',
                    'rule_name_ar': 'المجال العلائقي',
                    'rule_name_en': 'Relational Domain',
                    'threshold': relational_limit,
                    'threshold_unit': '%',
                    'actual': round(relational_percentage, 1),
                    'actual_unit': '%',
                    'enabled': True
                })
        
        # ============================================================
        # Check severity-specific thresholds (ABSOLUTE COUNTS)
        # ============================================================
        
        # Low Severity
        if enable_low_rule:
            low_limit = policy_snapshot.get('LowSeverityLimit', 0)
            low_count = domain_totals.get('low_severity_count', 0)
            
            if low_count > low_limit:
                violations.append({
                    'rule': 'LowSeverityLimit',
                    'rule_name_ar': 'الحالات منخفضة الخطورة',
                    'rule_name_en': 'Low Severity Cases',
                    'threshold': low_limit,
                    'threshold_unit': 'cases',
                    'actual': low_count,
                    'actual_unit': 'cases',
                    'enabled': True
                })
        
        # Medium Severity
        if enable_medium_rule:
            medium_limit = policy_snapshot.get('MediumSeverityLimit', 0)
            medium_count = domain_totals.get('medium_severity_count', 0)
            
            if medium_count > medium_limit:
                violations.append({
                    'rule': 'MediumSeverityLimit',
                    'rule_name_ar': 'الحالات متوسطة الخطورة',
                    'rule_name_en': 'Medium Severity Cases',
                    'threshold': medium_limit,
                    'threshold_unit': 'cases',
                    'actual': medium_count,
                    'actual_unit': 'cases',
                    'enabled': True
                })
        
        # High Severity
        if enable_high_rule:
            high_limit = policy_snapshot.get('HighSeverityLimit', 0)
            high_count = domain_totals.get('high_severity_count', 0)
            
            if high_count > high_limit:
                violations.append({
                    'rule': 'HighSeverityLimit',
                    'rule_name_ar': 'الحالات عالية الخطورة',
                    'rule_name_en': 'High Severity Cases',
                    'threshold': high_limit,
                    'threshold_unit': 'cases',
                    'actual': high_count,
                    'actual_unit': 'cases',
                    'enabled': True
                })
        
        is_compliant = len(violations) == 0
        
        # Determine explanation status based on compliance
        # FIXED: Correct status ID assignment
        # - Compliant reports: Status 4 = "No Explanation Needed"
        # - Non-compliant reports: Status 1 = "Waiting" (for explanation)
        # - After explanation submitted: Status 2 = "Responded"
        explanation_status_id = 4 if is_compliant else 1
        
        # Convert to JSON with ensure_ascii=False to preserve Arabic characters
        violated_rules_json = json.dumps(violations, ensure_ascii=False) if violations else None
        
        return {
            'is_compliant': is_compliant,
            'violated_rules': violated_rules_json,
            'explanation_status_id': explanation_status_id
        }
