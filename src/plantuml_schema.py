"""
PlantUML Star Schema Generator
================================
Generates a PlantUML (.puml) file depicting the MedSynora DW
star schema with actual column names from the dataset.
"""

import os


PLANTUML_CODE = r"""@startuml MedSynora_DW_Star_Schema

!define TABLE(x) entity x << (T,#FFAAAA) >>
!define FACT(x) entity x << (F,#FF7700) >>
!define DIM(x) entity x << (D,#AACCFF) >>
!define BRIDGE(x) entity x << (B,#AAFFAA) >>

skinparam entity {
    BackgroundColor<<F>> #FFF3E0
    BackgroundColor<<D>> #E3F2FD
    BackgroundColor<<B>> #E8F5E9
    BorderColor #333333
    FontSize 11
}

skinparam linetype ortho

title MedSynora DW — Star Schema Architecture
header Healthcare Data Warehouse for Patient Stratification

' ═══════════════════════════════════════════
' CENTRAL FACT TABLE
' ═══════════════════════════════════════════
FACT(FactEncounter) {
    * **Encounter_ID** : INT <<PK>>
    --
    Patient_ID : VARCHAR <<FK>>
    Disease_ID : INT <<FK>>
    ResponsibleDoctorID : INT <<FK>>
    InsuranceKey : INT <<FK>>
    RoomKey : INT <<FK>>
    CheckinDateKey : INT <<FK>>
    CheckoutDateKey : INT <<FK>>
    --
    Patient_Severity_Score : FLOAT
    RadiologyType : VARCHAR
    RadiologyProcedureCount : INT
    EndoscopyType : VARCHAR
    EndoscopyProcedureCount : INT
    CompanionPresent : BOOLEAN
}

' ═══════════════════════════════════════════
' SUPPORTING FACT TABLES
' ═══════════════════════════════════════════
FACT(FactVitals) {
    * **Encounter_ID** : INT <<FK>>
    --
    Patient_ID : VARCHAR
    Phase : VARCHAR
    HeartRate : FLOAT
    Temperature : FLOAT
    SystolicBP : INT
    DiastolicBP : INT
    RespRate : INT
    O2Sat : FLOAT
}

FACT(FactCost) {
    * **Encounter_ID** : INT <<FK>>
    --
    CostType : VARCHAR
    CostAmount : FLOAT
}

FACT(FactTreatment) {
    * **Encounter_ID** : INT <<FK>>
    --
    Treatment_ID : INT <<FK>>
    TreatmentDate : DATE
    TreatmentCost : FLOAT
    FollowUpRequired : BOOLEAN
}

FACT(FactLabTests) {
    * **Encounter_ID** : INT <<FK>>
    --
    TestName : VARCHAR
    TestResult : FLOAT
    TestUnit : VARCHAR
    NormalRange : VARCHAR
}

FACT(FactSpecialTests) {
    * **Encounter_ID** : INT <<FK>>
    --
    SpecialTest_ID : INT <<FK>>
    TestResult : VARCHAR
}

' ═══════════════════════════════════════════
' DIMENSION TABLES
' ═══════════════════════════════════════════
DIM(DimPatient) {
    * **Patient_ID** : VARCHAR <<PK>>
    --
    First_Name : VARCHAR
    Last_Name : VARCHAR
    Gender : VARCHAR
    Birth_Date : DATE
    Height : FLOAT
    Weight : FLOAT
    Marital_Status : VARCHAR
    Nationality : VARCHAR
    Blood_Type : VARCHAR
}

DIM(DimDisease) {
    * **Disease_ID** : INT <<PK>>
    --
    Admission_Diagnosis : VARCHAR
    Disease_Type : VARCHAR
    Disease_Severity : INT
    Medical_Unit : VARCHAR
}

DIM(DimDoctor) {
    * **Doctor_ID** : INT <<PK>>
    --
    Doctor_Name : VARCHAR
    Doctor_Surname : VARCHAR
    Doctor_Title : VARCHAR
    Doctor_Nationality : VARCHAR
    Medical_Unit : VARCHAR
    Max_Patient_Count : INT
}

DIM(DimInsurance) {
    * **InsuranceKey** : INT <<PK>>
    --
    Insurance_Plan_Name : VARCHAR
    Coverage_Limit : FLOAT
    Deductible : FLOAT
    Excluded_Treatments : VARCHAR
    Partial_Coverage : VARCHAR
}

DIM(DimRoom) {
    * **RoomKey** : INT <<PK>>
    --
    Care_Level : VARCHAR
    Room_Type : VARCHAR
}

DIM(DimDate) {
    * **DateKey** : INT <<PK>>
    --
    Date : DATE
    Year : INT
    Month : INT
    Day : INT
    Quarter : INT
    Weekday : INT
    Date_String : VARCHAR
}

DIM(DimTreatment) {
    * **Treatment_ID** : INT <<PK>>
    --
    Treatment_Name : VARCHAR
    Treatment_Category : VARCHAR
}

' ═══════════════════════════════════════════
' BRIDGE TABLE
' ═══════════════════════════════════════════
BRIDGE(BridgeEncounterDoctor) {
    Encounter_ID : INT <<FK>>
    Doctor_ID : INT <<FK>>
}

' ═══════════════════════════════════════════
' RELATIONSHIPS
' ═══════════════════════════════════════════
FactEncounter }|--|| DimPatient : "Patient_ID"
FactEncounter }|--|| DimDisease : "Disease_ID"
FactEncounter }|--|| DimInsurance : "InsuranceKey"
FactEncounter }|--|| DimRoom : "RoomKey"
FactEncounter }|--|| DimDate : "CheckinDateKey"

FactVitals }|--|| FactEncounter : "Encounter_ID"
FactCost }|--|| FactEncounter : "Encounter_ID"
FactTreatment }|--|| FactEncounter : "Encounter_ID"
FactLabTests }|--|| FactEncounter : "Encounter_ID"
FactSpecialTests }|--|| FactEncounter : "Encounter_ID"

BridgeEncounterDoctor }|--|| FactEncounter : "Encounter_ID"
BridgeEncounterDoctor }|--|| DimDoctor : "Doctor_ID"

FactTreatment }|--|| DimTreatment : "Treatment_ID"

@enduml
"""


def generate_plantuml(output_dir: str = "output/plantuml"):
    """Write the PlantUML star schema to a .puml file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "star_schema.puml")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(PLANTUML_CODE)

    print(f"\n[PlantUML] Star schema diagram saved to {filepath}")
    print("  -> Render at https://www.plantuml.com/plantuml/uml/ or via VS Code PlantUML extension")
    return filepath
