import pandas as pd
import os

# Carregar o arquivo CSV
df = pd.read_csv('/home/mpsilva-lx/Documentos/Detection-Chagas-Disease/code15_exams_nao_encontrados.csv')  # Substitua pelo nome do seu arquivo

print("="*80)
print("ANÁLISE DE EXAMS_PART NO DATASET")
print("="*80)

print(f"\n📊 ESTATÍSTICAS GERAIS:")
print(f"   Total de registros: {len(df)}")

# 1. Extrair o número da part do trace_file
df['exams_part'] = df['trace_file'].str.extract(r'exams_part(\d+)\.').astype(str)

# 2. Contar quantos registros por exams_part
contagem_por_part = df['exams_part'].value_counts().sort_index()

print(f"\n📁 DISTRIBUIÇÃO POR EXAMS_PART:")
print("-" * 60)
print(f"{'Exams_Part':<15} {'Quantidade':<15} {'Percentual':<15}")
print("-" * 60)

for part, count in contagem_por_part.items():
    percentual = (count / len(df)) * 100
    print(f"exams_part{part:<9} {count:<15} {percentual:.2f}%")

# 3. Identificar todas as parts possíveis (de 0 a 13, baseado no padrão CODE-15%)
# O CODE-15% tem exams_part de 0 a 13 (total 14 parts)
todas_parts_possiveis = set(str(i) for i in range(14))  # 0 a 13
parts_presentes = set(df['exams_part'].unique())
parts_faltantes = todas_parts_possiveis - parts_presentes

print(f"\n🔍 ANÁLISE DE EXAMS_PART FALTANTES:")
print("-" * 60)

print(f"\n   ✅ Parts presentes: {len(parts_presentes)}")
print(f"   📌 Parts faltantes: {len(parts_faltantes)}")

if len(parts_faltantes) > 0:
    print(f"\n   ⚠️  EXAMS_PART FALTANTES:")
    for part in sorted(parts_faltantes):
        print(f"      exams_part{part}.hdf5")
else:
    print(f"\n   ✅ Todas as exams_part (0 a 13) estão presentes!")

# 4. Estatísticas detalhadas
print(f"\n📈 ESTATÍSTICAS DETALHADAS:")
print("-" * 60)

print(f"\n   Média de registros por part: {contagem_por_part.mean():.2f}")
print(f"   Mediana de registros por part: {contagem_por_part.median():.2f}")
print(f"   Desvio padrão: {contagem_por_part.std():.2f}")
print(f"   Mínimo de registros em uma part: {contagem_por_part.min()}")
print(f"   Máximo de registros em uma part: {contagem_por_part.max()}")

# 5. Identificar parts com poucos registros (outliers)
media = contagem_por_part.mean()
desvio = contagem_por_part.std()
limite_inferior = media - desvio

parts_com_poucos_registros = contagem_por_part[contagem_por_part < limite_inferior]

if len(parts_com_poucos_registros) > 0:
    print(f"\n⚠️  EXAMS_PART COM POUCOS REGISTROS (abaixo de {limite_inferior:.2f}):")
    for part, count in parts_com_poucos_registros.items():
        print(f"      exams_part{part}: {count} registros")

# 6. Criar DataFrame com o resumo
resumo_df = pd.DataFrame({
    'exams_part': contagem_por_part.index,
    'quantidade': contagem_por_part.values,
    'percentual': (contagem_por_part.values / len(df) * 100).round(2)
})

# 7. Salvar resultados em CSV
resumo_df.to_csv('resumo_exams_part.csv', index=False)
print(f"\n✅ Resumo salvo em: resumo_exams_part.csv")

# 8. Gerar lista de arquivos faltantes
if len(parts_faltantes) > 0:
    faltantes_df = pd.DataFrame({
        'exams_part': sorted(parts_faltantes),
        'arquivo': [f'exams_part{part}.hdf5' for part in sorted(parts_faltantes)]
    })
    faltantes_df.to_csv('exams_part_faltantes.csv', index=False)
    print(f"✅ Lista de parts faltantes salva em: exams_part_faltantes.csv")

# 9. Gráfico ASCII da distribuição
print(f"\n📊 GRÁFICO DE DISTRIBUIÇÃO:")
print("-" * 60)

max_count = contagem_por_part.max()
escala = 50 / max_count  # 50 caracteres para o maior valor

for part in sorted(contagem_por_part.index):
    count = contagem_por_part[part]
    barra = '█' * int(count * escala)
    print(f"exams_part{part:<3} | {barra} {count}")

# 10. Análise por chagas_label (se existir a coluna)
if 'chagas' in df.columns:
    print(f"\n🩺 ANÁLISE POR CHAGAS POR EXAMS_PART:")
    print("-" * 60)
    
    # Tabela cruzada
    chagas_por_part = pd.crosstab(
        df['exams_part'], 
        df['chagas'], 
        margins=True, 
        margins_name='Total'
    )
    
    print("\n   Distribuição de Chagas por exams_part:")
    print(chagas_por_part.to_string())
    
    # Salvar análise de chagas
    chagas_por_part.to_csv('chagas_por_exams_part.csv')
    print(f"\n✅ Análise de Chagas salva em: chagas_por_exams_part.csv")

# 11. RESUMO EXECUTIVO
print(f"\n" + "="*80)
print("📊 RESUMO EXECUTIVO")
print("="*80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│                    ANÁLISE DE EXAMS_PART                        │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📁 TOTAL DE REGISTROS: {len(df):>6}                                       │
│   📁 TOTAL DE EXAMS_PART: {len(parts_presentes):>6} (de 0 a 13)                     │
│                                                                  │
│   ✅ PARTS PRESENTES: {len(parts_presentes):>6}                                   │
│   📌 PARTS FALTANTES: {len(parts_faltantes):>6}                                   │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📊 DISTRIBUIÇÃO:                                              │
│   • Média por part: {contagem_por_part.mean():.1f} registros                         │
│   • Part com mais registros: exams_part{contagem_por_part.idxmax()} ({contagem_por_part.max()} registros) │
│   • Part com menos registros: exams_part{contagem_por_part.idxmin()} ({contagem_por_part.min()} registros) │
│                                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📁 ARQUIVOS GERADOS:                                          │
│   1. resumo_exams_part.csv - Distribuição completa              │
│   2. exams_part_faltantes.csv - Lista de parts faltantes        │
""" + ("   3. chagas_por_exams_part.csv - Análise por Chagas\n" if 'chagas' in df.columns else "") + """
└────────────────────────────────────────────────────────────────┘
""")

# 12. Mostrar exemplos de cada part
print(f"\n📋 EXEMPLOS DE REGISTROS POR EXAMS_PART:")
print("-" * 60)

for part in sorted(parts_presentes)[:3]:  # Mostrar apenas 3 parts como exemplo
    exemplos = df[df['exams_part'] == part].head(2)
    print(f"\n   exams_part{part} (total: {contagem_por_part[part]} registros):")
    print(f"   Primeiros 2 exam_id: {', '.join(exemplos['exam_id'].astype(str).tolist())}")