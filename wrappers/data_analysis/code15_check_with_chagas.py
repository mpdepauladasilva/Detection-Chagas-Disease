import pandas as pd

# Carregar os arquivos CSV
chagas_labels_df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/datasets_original_csv/code15_chagas_labels.csv')
exams_original_df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/datasets_original_csv/code15_exams_original.csv')

# Garantir que exam_id e patient_id sejam strings para comparação
chagas_labels_df['exam_id'] = chagas_labels_df['exam_id'].astype(str)
chagas_labels_df['patient_id'] = chagas_labels_df['patient_id'].astype(str)

exams_original_df['exam_id'] = exams_original_df['exam_id'].astype(str)
exams_original_df['patient_id'] = exams_original_df['patient_id'].astype(str)

print("="*80)
print("COMPARAÇÃO ENTRE OS DATASETS")
print("="*80)

print(f"\n📊 ESTATÍSTICAS GERAIS:")
print(f"   code15_chagas_labels.csv: {len(chagas_labels_df)} registros")
print(f"   code15_exams_original.csv: {len(exams_original_df)} registros")

# Criar conjuntos de exam_id e patient_id
chagas_exams = set(chagas_labels_df['exam_id'])
chagas_patients = set(chagas_labels_df['patient_id'])

exams_original_exams = set(exams_original_df['exam_id'])
exams_original_patients = set(exams_original_df['patient_id'])

print(f"\n🔍 ANÁLISE POR EXAM_ID:")
print("-" * 40)

# Examinar exam_id presentes em ambos
exams_em_comum = chagas_exams.intersection(exams_original_exams)
exams_so_no_chagas = chagas_exams - exams_original_exams
exams_so_no_original = exams_original_exams - chagas_exams

print(f"   ✅ Exam_id em ambos: {len(exams_em_comum)}")
print(f"   📌 Exam_id só no chagas_labels: {len(exams_so_no_chagas)}")
print(f"   📌 Exam_id só no exams_original: {len(exams_so_no_original)}")

print(f"\n🔍 ANÁLISE POR PATIENT_ID:")
print("-" * 40)

# Examinar patient_id presentes em ambos
patients_em_comum = chagas_patients.intersection(exams_original_patients)
patients_so_no_chagas = chagas_patients - exams_original_patients
patients_so_no_original = exams_original_patients - chagas_patients

print(f"   ✅ Patient_id em ambos: {len(patients_em_comum)}")
print(f"   📌 Patient_id só no chagas_labels: {len(patients_so_no_chagas)}")
print(f"   📌 Patient_id só no exams_original: {len(patients_so_no_original)}")

# 1. CRIAR DATASET COM REGISTROS QUE ESTÃO EM AMBOS
print(f"\n📁 CRIANDO ARQUIVOS DE COMPARAÇÃO:")
print("-" * 40)

# Registros que estão em ambos (usando exam_id como chave)
em_ambos_df = pd.merge(
    chagas_labels_df,
    exams_original_df,
    on=['exam_id', 'patient_id'],
    how='inner'
)

print(f"\n   ✅ Registros em ambos (exam_id + patient_id iguais): {len(em_ambos_df)}")
em_ambos_df.to_csv('registros_em_ambos.csv', index=False)

# 2. REGISTROS QUE NÃO ESTÃO EM AMBOS

# 2.1 Registros que estão no chagas_labels mas NÃO no exams_original
so_no_chagas_df = chagas_labels_df[
    ~chagas_labels_df['exam_id'].isin(exams_original_exams)
].copy()
so_no_chagas_df['motivo'] = 'Não encontrado no exams_original'

print(f"\n   📌 Registros só no chagas_labels: {len(so_no_chagas_df)}")
if len(so_no_chagas_df) > 0:
    so_no_chagas_df.to_csv('registros_so_no_chagas.csv', index=False)
    print(f"      - Primeiros 5 exemplos:")
    print(so_no_chagas_df[['exam_id', 'patient_id', 'chagas']].head().to_string())

# 2.2 Registros que estão no exams_original mas NÃO no chagas_labels
so_no_original_df = exams_original_df[
    ~exams_original_df['exam_id'].isin(chagas_exams)
].copy()
so_no_original_df['motivo'] = 'Não encontrado no chagas_labels'

print(f"\n   📌 Registros só no exams_original: {len(so_no_original_df)}")
if len(so_no_original_df) > 0:
    so_no_original_df.to_csv('registros_so_no_original.csv', index=False)
    print(f"      - Primeiros 5 exemplos:")
    print(so_no_original_df[['exam_id', 'patient_id', 'age', 'is_male']].head().to_string())

# 3. ANÁLISE DE INCONSISTÊNCIAS
print(f"\n🔍 ANÁLISE DE INCONSISTÊNCIAS:")
print("-" * 40)

# Registrar exam_id que existem em ambos mas com patient_id diferentes
merged_check = pd.merge(
    chagas_labels_df,
    exams_original_df,
    on='exam_id',
    how='inner',
    suffixes=('_chagas', '_original')
)

# Verificar onde patient_id é diferente
patient_id_diferentes = merged_check[
    merged_check['patient_id_chagas'] != merged_check['patient_id_original']
]

print(f"\n   ⚠️  Exam_id com patient_id diferentes entre os arquivos: {len(patient_id_diferentes)}")

if len(patient_id_diferentes) > 0:
    inconsistencias_df = patient_id_diferentes[[
        'exam_id', 
        'patient_id_chagas', 
        'patient_id_original',
        'chagas'
    ]].copy()
    inconsistencias_df.to_csv('inconsistencias_patient_id.csv', index=False)
    print(f"      - Lista salva em: inconsistencias_patient_id.csv")
    print(f"      - Primeiros exemplos:")
    print(inconsistencias_df.head().to_string())

# 4. RESUMO EXECUTIVO
print(f"\n" + "="*80)
print("📊 RESUMO EXECUTIVO")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│                      RESUMO DA COMPARAÇÃO                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📁 ARQUIVO 1: code15_chagas_labels.csv                        │
│   📁 ARQUIVO 2: code15_exams_original.csv                       │
│                                                                  │
│   ✅ REGISTROS EM COMUM (exam_id + patient_id): {len(em_ambos_df):>6}             │
│                                                                  │
│   📌 SÓ NO CHAGAS_LABELS: {len(so_no_chagas_df):>6} registros                   │
│   📌 SÓ NO EXAMS_ORIGINAL: {len(so_no_original_df):>6} registros                │
│                                                                  │
│   ⚠️  INCONSISTÊNCIAS (patient_id diferente): {len(patient_id_diferentes):>6}      │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📁 ARQUIVOS GERADOS:                                          │
│   1. registros_em_ambos.csv - {len(em_ambos_df):>6} registros válidos           │
│   2. registros_so_no_chagas.csv - {len(so_no_chagas_df):>6} registros           │
│   3. registros_so_no_original.csv - {len(so_no_original_df):>6} registros       │
│   4. inconsistencias_patient_id.csv - {len(patient_id_diferentes):>6} registros │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
""")

# 5. MOSTRAR EXEMPLOS DE CADA CATEGORIA (CORRIGIDO)
print(f"\n📋 EXEMPLOS DE CADA CATEGORIA:")
print("-" * 40)

if len(em_ambos_df) > 0:
    print(f"\n✅ Exemplos de registros em ambos:")
    # CORREÇÃO: Usar os nomes corretos das colunas
    if 'patient_id' in em_ambos_df.columns:
        print(em_ambos_df[['exam_id', 'patient_id']].head(3).to_string())
    else:
        print(em_ambos_df[['exam_id']].head(3).to_string())

if len(so_no_chagas_df) > 0:
    print(f"\n📌 Exemplos de registros só no chagas_labels:")
    print(so_no_chagas_df[['exam_id', 'patient_id', 'chagas']].head(3).to_string())

if len(so_no_original_df) > 0:
    print(f"\n📌 Exemplos de registros só no exams_original:")
    print(so_no_original_df[['exam_id', 'patient_id', 'age', 'is_male']].head(3).to_string())

if len(patient_id_diferentes) > 0:
    print(f"\n⚠️  Exemplos de inconsistências (patient_id diferente):")
    # CORREÇÃO: Verificar se as colunas existem
    cols_to_show = ['exam_id', 'patient_id_chagas', 'patient_id_original']
    available_cols = [col for col in cols_to_show if col in patient_id_diferentes.columns]
    if available_cols:
        print(patient_id_diferentes[available_cols].head(3).to_string())

# 6. ESTATÍSTICAS ADICIONAIS
print(f"\n📈 ESTATÍSTICAS ADICIONAIS:")
print("-" * 40)

# Distribuição de chagas nos registros em comum
if len(em_ambos_df) > 0 and 'chagas' in em_ambos_df.columns:
    print(f"\n   Distribuição de chagas nos registros em comum:")
    print(em_ambos_df['chagas'].value_counts())

# Distribuição de chagas nos registros só no chagas_labels
if len(so_no_chagas_df) > 0:
    print(f"\n   Distribuição de chagas nos registros exclusivos do chagas_labels:")
    print(so_no_chagas_df['chagas'].value_counts())

# Taxa de correspondência
taxa_correspondencia = (len(em_ambos_df) / len(chagas_labels_df)) * 100 if len(chagas_labels_df) > 0 else 0
print(f"\n   📊 Taxa de correspondência (chagas_labels em relação ao original): {taxa_correspondencia:.2f}%")