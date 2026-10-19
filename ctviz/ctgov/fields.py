"""ClinicalTrials.gov field names: projection sets (piece names) and canonical JSON paths."""

from __future__ import annotations

from enum import StrEnum


class FieldSet(StrEnum):
    LANDSCAPE = "landscape"
    OUTCOMES = "outcomes"
    PEOPLE = "people"
    ELIGIBILITY = "eligibility"
    FULL = "full"  # no projection: GET /studies/{nct} or full search records


_LANDSCAPE = [
    "NCTId",
    "BriefTitle",
    "Acronym",
    "OverallStatus",
    "WhyStopped",
    "Phase",
    "StudyType",
    "StartDate",
    "StartDateType",
    "PrimaryCompletionDate",
    "PrimaryCompletionDateType",
    "CompletionDate",
    "CompletionDateType",
    "LastUpdatePostDate",
    "LeadSponsorName",
    "LeadSponsorClass",
    "CollaboratorName",
    "CollaboratorClass",
    "Condition",
    "ConditionMeshTerm",
    "InterventionName",
    "InterventionType",
    "InterventionOtherName",
    "InterventionMeshTerm",
    "ArmGroupLabel",
    "ArmGroupType",
    "ArmGroupInterventionName",
    "EnrollmentCount",
    "EnrollmentType",
    "LocationCountry",
    "DesignAllocation",
    "DesignInterventionModel",
    "DesignPrimaryPurpose",
    "DesignMasking",
    "DesignWhoMasked",
    "HasResults",
    "StdAge",
    "Sex",
]

FIELD_SETS: dict[FieldSet, list[str] | None] = {
    FieldSet.LANDSCAPE: _LANDSCAPE,
    FieldSet.OUTCOMES: _LANDSCAPE
    + [
        "PrimaryOutcomeMeasure",
        "PrimaryOutcomeDescription",
        "PrimaryOutcomeTimeFrame",
        "SecondaryOutcomeMeasure",
        "SecondaryOutcomeTimeFrame",
    ],
    FieldSet.PEOPLE: _LANDSCAPE
    + [
        "OverallOfficialName",
        "OverallOfficialAffiliation",
        "OverallOfficialRole",
        "LocationFacility",
        "LocationCity",
        "LocationState",
        "LocationStatus",
        "LocationGeoPoint",
    ],
    FieldSet.ELIGIBILITY: _LANDSCAPE
    + [
        "EligibilityCriteria",
        "MinimumAge",
        "MaximumAge",
        "HealthyVolunteers",
        "BriefSummary",
    ],
    FieldSet.FULL: None,
}

# Canonical paths used by compute code (so citations always use identical field_paths).
P = {
    "nct_id": "protocolSection.identificationModule.nctId",
    "brief_title": "protocolSection.identificationModule.briefTitle",
    "official_title": "protocolSection.identificationModule.officialTitle",
    "acronym": "protocolSection.identificationModule.acronym",
    "status": "protocolSection.statusModule.overallStatus",
    "why_stopped": "protocolSection.statusModule.whyStopped",
    "start_date": "protocolSection.statusModule.startDateStruct.date",
    "start_date_type": "protocolSection.statusModule.startDateStruct.type",
    "primary_completion_date": "protocolSection.statusModule.primaryCompletionDateStruct.date",
    "primary_completion_type": "protocolSection.statusModule.primaryCompletionDateStruct.type",
    "completion_date": "protocolSection.statusModule.completionDateStruct.date",
    "completion_type": "protocolSection.statusModule.completionDateStruct.type",
    "first_post_date": "protocolSection.statusModule.studyFirstPostDateStruct.date",
    "results_first_post_date": "protocolSection.statusModule.resultsFirstPostDateStruct.date",
    "last_update_date": "protocolSection.statusModule.lastUpdatePostDateStruct.date",
    "lead_sponsor": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "lead_sponsor_class": "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
    "collaborators": "protocolSection.sponsorCollaboratorsModule.collaborators",
    "brief_summary": "protocolSection.descriptionModule.briefSummary",
    "detailed_description": "protocolSection.descriptionModule.detailedDescription",
    "conditions": "protocolSection.conditionsModule.conditions",
    "keywords": "protocolSection.conditionsModule.keywords",
    "study_type": "protocolSection.designModule.studyType",
    "phases": "protocolSection.designModule.phases",
    "allocation": "protocolSection.designModule.designInfo.allocation",
    "intervention_model": "protocolSection.designModule.designInfo.interventionModel",
    "primary_purpose": "protocolSection.designModule.designInfo.primaryPurpose",
    "masking": "protocolSection.designModule.designInfo.maskingInfo.masking",
    "who_masked": "protocolSection.designModule.designInfo.maskingInfo.whoMasked",
    "enrollment_count": "protocolSection.designModule.enrollmentInfo.count",
    "enrollment_type": "protocolSection.designModule.enrollmentInfo.type",
    "arm_groups": "protocolSection.armsInterventionsModule.armGroups",
    "interventions": "protocolSection.armsInterventionsModule.interventions",
    "primary_outcomes": "protocolSection.outcomesModule.primaryOutcomes",
    "secondary_outcomes": "protocolSection.outcomesModule.secondaryOutcomes",
    "eligibility_criteria": "protocolSection.eligibilityModule.eligibilityCriteria",
    "healthy_volunteers": "protocolSection.eligibilityModule.healthyVolunteers",
    "sex": "protocolSection.eligibilityModule.sex",
    "min_age": "protocolSection.eligibilityModule.minimumAge",
    "max_age": "protocolSection.eligibilityModule.maximumAge",
    "std_ages": "protocolSection.eligibilityModule.stdAges",
    "central_contacts": "protocolSection.contactsLocationsModule.centralContacts",
    "overall_officials": "protocolSection.contactsLocationsModule.overallOfficials",
    "locations": "protocolSection.contactsLocationsModule.locations",
    "references": "protocolSection.referencesModule.references",
    "condition_meshes": "derivedSection.conditionBrowseModule.meshes",
    "intervention_meshes": "derivedSection.interventionBrowseModule.meshes",
    "has_results": "hasResults",
    "participant_flow": "resultsSection.participantFlowModule",
    "baseline": "resultsSection.baselineCharacteristicsModule",
    "outcome_measures": "resultsSection.outcomeMeasuresModule.outcomeMeasures",
    "adverse_events": "resultsSection.adverseEventsModule",
    "limitations": "resultsSection.moreInfoModule.limitationsAndCaveats.description",
}

PHASE_ORDER = ["EARLY_PHASE1", "PHASE1", "PHASE1/PHASE2", "PHASE2", "PHASE2/PHASE3", "PHASE3", "PHASE4", "NA"]
STATUS_ORDER = [
    "NOT_YET_RECRUITING",
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "COMPLETED",
    "TERMINATED",
    "WITHDRAWN",
    "UNKNOWN",
]
# Status today, grouped. Terminated and withdrawn trials are over, so they count as done; suspended trials
# are paused (neither), unknown/expanded-access records are neither too.
STATUS_GROUPS: dict[str, tuple[str, ...]] = {
    "ACTIVE": ("NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"),
    "DONE": ("COMPLETED", "TERMINATED", "WITHDRAWN"),
    "OTHER": (
        "SUSPENDED",
        "UNKNOWN",
        "AVAILABLE",
        "NO_LONGER_AVAILABLE",
        "TEMPORARILY_NOT_AVAILABLE",
        "APPROVED_FOR_MARKETING",
        "WITHHELD",
    ),
}
STATUS_GROUP_LABELS = {
    "ACTIVE": "Active",
    "DONE": "Done (completed, terminated, withdrawn)",
    "OTHER": "Other (suspended, unknown, expanded access)",
}


def status_group(status: str | None) -> str:
    return next((g for g, members in STATUS_GROUPS.items() if status in members), "OTHER")


def status_phrase(statuses: list[str] | tuple[str, ...]) -> str | None:
    """How a status filter reads in titles and summaries: always "status today", never "were recruiting"."""
    if not statuses:
        return None
    chosen = set(statuses)
    groups = [g for g, members in STATUS_GROUPS.items() if chosen == set(members)]
    if groups:
        words = {"ACTIVE": "active", "DONE": "completed, terminated or withdrawn", "OTHER": "other"}[groups[0]]
    else:
        names = [s.replace("_", " ").lower().replace("active not", "active, not") for s in statuses]
        words = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
    return f"status today: {words}"


PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE1/PHASE2": "Phase 1/2",
    "PHASE2": "Phase 2",
    "PHASE2/PHASE3": "Phase 2/3",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not applicable",
}
