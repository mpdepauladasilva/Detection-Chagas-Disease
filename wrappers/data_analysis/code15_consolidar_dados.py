import pandas as pd


class Code15DatasetProcessor:

    def __init__(self, chagas_path: str, code15_path: str):
        self.chagas_path = chagas_path
        self.code15_path = code15_path

    def load_datasets(self):

        self.chagas_df = pd.read_csv(self.chagas_path, low_memory=False)
        self.code15_df = pd.read_csv(self.code15_path, low_memory=False)

        print("\nDatasets carregados:")
        print("Chagas:", len(self.chagas_df))
        print("CODE15:", len(self.code15_df))

    def prepare_keys(self):

        self.chagas_df["record_id"] = self.chagas_df["record_id"].astype(str)
        self.code15_df["exam_id"] = self.code15_df["exam_id"].astype(str)

    def merge_datasets(self):

        merged = pd.merge(
            self.chagas_df,
            self.code15_df[
                ["exam_id", "patient_id", "death", "normal_ecg"]
            ],
            left_on="record_id",
            right_on="exam_id",
            how="left",
        )

        merged.drop("exam_id", axis=1, inplace=True)

        # condição obrigatória
        merged = merged[
            merged["patient_id"].notna()
        ].copy()

        self.merged_df = merged

        print("\n1. MERGE REALIZADO")
        print("Registros após filtro patient_id:", len(merged))

    def create_missing_flags(self):

        df = self.merged_df

        df["flag_death_faltante"] = df["death"].isna()
        df["flag_normal_ecg_faltante"] = df["normal_ecg"].isna()

        self.merged_df = df

    def export_main_dataset(self):

        df = self.merged_df[
            [
                "record_id",
                "patient_id",
                "dataset",
                "age",
                "sex",
                "chagas_label",
                "death",
                "normal_ecg",
                "file_path",
                "dat_path",
                "flag_death_faltante",
                "flag_normal_ecg_faltante",
            ]
        ].copy()

        df.rename(
            columns={
                "chagas_label": "chagas_label_absolut"
            },
            inplace=True
        )

        df["patient_id"] = df["patient_id"].astype("Int64")

        df.to_csv("dataset_code15_completo_flag.csv", index=False)

        print("\n2. DATASET PRINCIPAL GERADO")
        print("dataset_code15_completo_flag.csv")
        print("Registros:", len(df))

        self.dataset_final = df

    def export_exams_not_in_chagas(self):

        record_ids = set(self.chagas_df["record_id"])
        exam_ids = set(self.code15_df["exam_id"])

        missing = exam_ids - record_ids

        df_missing = self.code15_df[
            self.code15_df["exam_id"].isin(missing)
        ].copy()

        df_missing.to_csv(
            "code15_exams_nao_encontrados.csv",
            index=False
        )

        print("\n3. EXAMES DO CODE15 NÃO PRESENTES NO DATASET CHAGAS")
        print("Total:", len(df_missing))

    def print_statistics(self):

        df = self.dataset_final

        print("\n" + "=" * 80)
        print("ESTATÍSTICAS")
        print("=" * 80)

        print("\nDistribuição Chagas:")
        print(df["chagas_label_absolut"].value_counts())

        print("\nMissing death:")
        print(df["flag_death_faltante"].sum())

        print("\nMissing normal_ecg:")
        print(df["flag_normal_ecg_faltante"].sum())


def main():

    processor = Code15DatasetProcessor(
        "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv",
        "/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/registros_em_ambos.csv",
    )

    print("=" * 80)
    print("CONSOLIDAÇÃO DATASET CODE15")
    print("=" * 80)

    processor.load_datasets()
    processor.prepare_keys()
    processor.merge_datasets()
    processor.create_missing_flags()
    processor.export_main_dataset()
    processor.export_exams_not_in_chagas()
    processor.print_statistics()


if __name__ == "__main__":
    main()