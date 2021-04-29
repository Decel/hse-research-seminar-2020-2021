import pandas as pd

import plot
import predicates
import preprocessing

ADMINISTRATION_DATA_PATH = "data/administration.xlsx"
STUDENTS_DATA_PATH = "data/students.xlsx"
Q_ADMINISTRATION = "Gender"
Q_ADMINISTRATION_TITLE = "gender"
Q_STUDENT = "Используешь ли ты Wi-Fi в школе?"
Q_STUDENT_TITLE = "does_use_wi-fi"
OUT_ADMINISTRATION = f"img/administration"
OUT_STUDENTS = f"img/students"

EMBEDDINGS = [preprocessing.prepare_pca_dataframe, preprocessing.prepare_umap_dataframe]


def prepare_outfile_name(basename, algorithm, question_name):
    return f"{basename}_{algorithm}_{question_name}.png"


if __name__ == '__main__':
    for filename in [ADMINISTRATION_DATA_PATH, STUDENTS_DATA_PATH]:
        dataframe = pd.read_excel(filename, index_col=0, header=0)

        if filename == ADMINISTRATION_DATA_PATH:
            target_field_name = Q_ADMINISTRATION
            predicate_filter = predicates.predicate_administration
            basename = OUT_ADMINISTRATION
            question_name = Q_ADMINISTRATION_TITLE
        elif filename == STUDENTS_DATA_PATH:
            target_field_name = Q_STUDENT
            predicate_filter = predicates.predicate_students
            basename = OUT_STUDENTS
            question_name = Q_STUDENT_TITLE
        else:
            raise NotImplementedError()

        x_axis, _ = preprocessing.normalize_data_frames(dataframe, target_field_name, predicates.predicate_filter_other_questions)

        for embedding_func in EMBEDDINGS:
            if embedding_func == preprocessing.prepare_pca_dataframe:
                title = "PCA"
            elif embedding_func == preprocessing.prepare_umap_dataframe:
                title = "UMAP"
            else:
                raise NotImplementedError()

            out_filename = prepare_outfile_name(basename, title, question_name)
            embedded_frame = embedding_func(dataframe, x_axis, target_field_name)

            if filename == STUDENTS_DATA_PATH:
                embedded_frame[target_field_name] = embedded_frame[target_field_name].shift(-1)
                embedded_frame.drop(embedded_frame.tail(1).index, inplace=True)

            plot.plot_embedding(embedded_frame, target_field_name, predicate_filter, out_filename, question_name)
