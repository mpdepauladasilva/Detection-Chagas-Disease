import pandas as pd

# Carregar os arquivos CSV
primeiro_df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/dataset_chagas_absolut.csv')
segundo_df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/assets/datasets_original_csv/code15_chagas_labels.csv')

# Mostrar as colunas para confirmar
print("Colunas do primeiro arquivo (dataset_chagas_absolut.csv):")
print(primeiro_df.columns.tolist())
print("\nPrimeiras linhas do primeiro arquivo:")
print(primeiro_df.head(3))
print("\n" + "="*50)

print("\nColunas do segundo arquivo:")
print(segundo_df.columns.tolist())
print("\nPrimeiras linhas do segundo arquivo:")
print(segundo_df.head(3))
print("\n" + "="*50)

# Fazer o merge para encontrar o patient_id correspondente ao record_id
merged_df = pd.merge(
    primeiro_df,
    segundo_df[['exam_id', 'patient_id', 'chagas']],
    left_on='record_id',
    right_on='exam_id',
    how='left'
)

# Remover a coluna exam_id (duplicada do record_id)
merged_df.drop('exam_id', axis=1, inplace=True)

# Converter chagas_label (0/1) para boolean para comparação
merged_df['chagas_label_bool'] = merged_df['chagas_label'].map({0: False, 1: True})

# Verificar se a conversão foi feita corretamente
print("\nVerificação de conversão:")
print("Valores únicos em chagas_label (original):", merged_df['chagas_label'].unique())
print("Valores únicos em chagas_label_bool (convertido):", merged_df['chagas_label_bool'].unique())
print("Valores únicos em chagas (do segundo arquivo):", merged_df['chagas'].unique())

# Criar colunas para verificar compatibilidade
merged_df['possui_patient_id'] = merged_df['patient_id'].notna()
merged_df['chagas_compativel'] = (merged_df['chagas_label_bool'] == merged_df['chagas'])

# Separar os dados
# Dados válidos: possuem patient_id E chagas compatível
dados_validos = merged_df[
    merged_df['possui_patient_id'] & 
    merged_df['chagas_compativel']
].copy()

# Dados inválidos: não possuem patient_id OU chagas incompatível
dados_invalidos = merged_df[
    ~(merged_df['possui_patient_id'] & merged_df['chagas_compativel'])
].copy()

# Remover colunas auxiliares dos dados válidos antes de salvar
dados_validos.drop(['chagas_label_bool', 'possui_patient_id', 'chagas_compativel', 'chagas'], 
                   axis=1, inplace=True, errors='ignore')

# Manter a coluna 'chagas' nos dados inválidos para análise
dados_invalidos.drop(['chagas_label_bool', 'possui_patient_id', 'chagas_compativel'], 
                     axis=1, inplace=True, errors='ignore')

# Reordenar colunas para colocar patient_id no início (após record_id)
colunas_ordenadas = ['record_id', 'patient_id'] + [col for col in dados_validos.columns 
                                                    if col not in ['record_id', 'patient_id']]
dados_validos = dados_validos[colunas_ordenadas]

# Salvar os arquivos
dados_validos.to_csv('dados_completos_validos.csv', index=False)
dados_invalidos.to_csv('dados_inconsistentes.csv', index=False)

# Mostrar estatísticas
print("\n" + "="*50)
print("RESULTADOS FINAIS:")
print("="*50)
print(f"Total de registros no primeiro arquivo: {len(primeiro_df)}")
print(f"Registros encontrados no segundo arquivo (com patient_id): {merged_df['patient_id'].notna().sum()}")
print(f"Registros com dados válidos (com patient_id e chagas compatível): {len(dados_validos)}")
print(f"Registros com inconsistências: {len(dados_invalidos)}")

# Análise detalhada das inconsistências
if len(dados_invalidos) > 0:
    print("\n" + "="*50)
    print("ANÁLISE DETALHADA DAS INCONSISTÊNCIAS:")
    print("="*50)
    
    # Separar por tipo de inconsistência
    sem_patient_id = dados_invalidos[dados_invalidos['patient_id'].isna()]
    com_patient_id_incompativel = dados_invalidos[dados_invalidos['patient_id'].notna()]
    
    print(f"\n1. Registros SEM patient_id correspondente: {len(sem_patient_id)}")
    if len(sem_patient_id) > 0:
        print("\n   Exemplos (primeiros 5):")
        print(sem_patient_id[['record_id', 'chagas_label', 'dataset']].head().to_string())
    
    print(f"\n2. Registros COM patient_id mas chagas incompatível: {len(com_patient_id_incompativel)}")
    if len(com_patient_id_incompativel) > 0:
        print("\n   Exemplos (primeiros 5):")
        # Para esses, mostrar a comparação
        com_patient_id_incompativel['chagas_label_bool_temp'] = com_patient_id_incompativel['chagas_label'].map({0: False, 1: True})
        exemplos = com_patient_id_incompativel.head(5)[
            ['record_id', 'patient_id', 'chagas_label', 'chagas_label_bool_temp', 'chagas']
        ]
        print(exemplos.to_string())
        com_patient_id_incompativel.drop('chagas_label_bool_temp', axis=1, inplace=True)

# Estatísticas de correspondência
print("\n" + "="*50)
print("ESTATÍSTICAS DE CORRESPONDÊNCIA:")
print("="*50)
print(f"Total de exames no primeiro arquivo: {len(primeiro_df)}")
print(f"Exames com patient_id encontrado: {merged_df['patient_id'].notna().sum()}")
print(f"Taxa de correspondência: {(merged_df['patient_id'].notna().sum()/len(primeiro_df)*100):.2f}%")

print(f"\nDistribuição de chagas_label no primeiro arquivo:")
print(primeiro_df['chagas_label'].value_counts().sort_index())

print(f"\nDistribuição de chagas no segundo arquivo (nos registros encontrados):")
registros_encontrados = merged_df[merged_df['patient_id'].notna()]
print(registros_encontrados['chagas'].value_counts())

print(f"\nArquivos gerados:")
print(f"✅ dados_completos_validos.csv - {len(dados_validos)} registros válidos")
print(f"✅ dados_inconsistentes.csv - {len(dados_invalidos)} registros com problemas")