import pandas as pd

METADATA_DIR = "../metadata/"

gender_df = pd.read_csv(f"{METADATA_DIR}/predicted_gender_nopii.csv", index_col=0)
ALL_SUBJECT_IDS = pd.read_csv(f"{METADATA_DIR}/subject_ids.txt", header=None)[
    0
].to_list()
EXPECTED_NUM_SUBJECTS = len(ALL_SUBJECT_IDS)


# Get Student Subject IDs
profession_df = pd.read_csv(f"{METADATA_DIR}/subject_professions.csv")
# print(profession_df.head())
STUDENT_SUBJECT_IDS_GROUNDTRUTH = profession_df.loc[
    profession_df["Title"].str.lower() == "student", "SubjectID"
].tolist()
STUDENT_SUBJECT_IDS_GROUNDTRUTH = sorted(
    list(set(STUDENT_SUBJECT_IDS_GROUNDTRUTH).intersection(set(ALL_SUBJECT_IDS)))
)  # Seems shorter than the actually reported number of students
STUDENT_SUBJECT_IDS = pd.read_csv(
    f"{METADATA_DIR}/student_subject_ids_robust.csv", header=None
)[
    1
].to_list()  # from robust clustering assignments

# Population-specific question ids
population_specific_question_ids = {
    "physicians": [
        "Q09",  # If you are a physician, did you train in the US at any point?
    ],
    "students": [
        "Q07",  # If student or trainee, what year are you in?
        "Q36",  # Has this crisis affected your specialty decision or career plans in any way?
        "Q39",  # If student or trainee, how closely do you feel that you are adhering to the Hippocratic oath during this time?
        "Q40",  # If student or trainee, do you agree with your school's policies regarding medical students' roles at this time?
    ],
    "nonstudents": [  # i.e. physicians, nurses, residents etc.
        "Q11",  # How long have you been practicing?
        "Q12",  # Over the past two months, have you practiced clinically in areas where you could be in touch with patients who have covid-19?
        "Q21",  # How has it changed your approach to management? (different from usual, at odds with existing guidelines, may not be as effective, etc.)
        "Q22",  # Are your processes different for end-of-life decisions? Do you have to take people off ventilator more frequently?
        "Q30",  # How do you feel about working from home OR at the frontlines?
        "Q31",  # Do you feel you should be able to handle this as a healthcare professional?
        "Q35",  # Would you change jobs or career trajectories?
    ],
}
population_specific_question_ids_all = sorted(
    [item for sublist in population_specific_question_ids.values() for item in sublist]
)
population_subject_ids = {
    "physicians": pd.read_csv(
        f"{METADATA_DIR}/physician_subject_ids_robust.csv", header=None
    )[1].to_list(),
    "students": STUDENT_SUBJECT_IDS,
    "nonstudents": list(set(ALL_SUBJECT_IDS) - set(STUDENT_SUBJECT_IDS)),
}


def get_all_subject_ids():
    # df = pd.read_csv(f"{METADATA_DIR}/subject_ids.txt")
    # df.columns = ["subject_id"]
    # return df["subject_id"].tolist()
    return ALL_SUBJECT_IDS


def get_predicted_gender(subject_id="C003"):
    if subject_id in gender_df.index:
        return gender_df.loc[subject_id, "PredictedGender"].strip()
    else:
        return "U"  # Unknown


def create_primary_cluster_subject_map(clusters_df):
    cluster_to_subjects = {}
    for index, row in clusters_df.iterrows():
        cluster_ids = str(row["cluster_ids"]).strip('"').strip("'").split(",")
        for cluster_id in cluster_ids:
            cluster_id = cluster_id.strip()
            if (
                cluster_id and cluster_id != "nan"
            ):  # Check if cluster_id is not an empty string or 'nan'
                if cluster_id not in cluster_to_subjects:
                    cluster_to_subjects[cluster_id] = []
                cluster_to_subjects[cluster_id].append(row["subject_id"])
    return cluster_to_subjects


def create_secondary_cluster_subject_map(clusters_df):
    cluster_to_subjects = {}
    for index, row in clusters_df.iterrows():
        cluster_ids = str(row["secondary_cluster_ids"]).strip('"').strip("'").split(",")
        for cluster_id in cluster_ids:
            cluster_id = cluster_id.strip()
            if (
                cluster_id and cluster_id != "nan"
            ):  # Check if cluster_id is not an empty string or 'nan'
                if cluster_id not in cluster_to_subjects:
                    cluster_to_subjects[cluster_id] = []
                cluster_to_subjects[cluster_id].append(row["subject_id"])
    return cluster_to_subjects


if __name__ == "__main__":
    all_subject_genders = [
        get_predicted_gender(subject_id=subject_id)
        for subject_id in get_all_subject_ids()
    ]
    print(pd.Series(all_subject_genders).value_counts())

    print("get_subject_ids: ", get_all_subject_ids())

    print(
        "get_predicted_gender(subject_id=C003): ",
        get_predicted_gender(subject_id="C003"),
    )
    print(
        [
            (subject_id, get_predicted_gender(subject_id=subject_id))
            for subject_id in get_all_subject_ids()
        ]
    )
    print(
        "STUDENT_SUBJECT_IDS_GROUNDTRUTH: ",
        STUDENT_SUBJECT_IDS_GROUNDTRUTH,
        len(STUDENT_SUBJECT_IDS_GROUNDTRUTH),
    )
    print("STUDENT_SUBJECT_IDS: ", STUDENT_SUBJECT_IDS, len(STUDENT_SUBJECT_IDS))

    print("populations: ", population_subject_ids.keys())

    print("All population_specific_question_ids:", population_specific_question_ids_all)
