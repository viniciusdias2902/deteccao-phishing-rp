"""Pipeline de análise comparativa de classificadores para detecção de phishing em URLs."""

import json
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})
sns.set_style('whitegrid')

NOMES = [
    'Regressão Logística',
    'Naive Bayes',
    'KNN (k=5)',
    'Árvore de Decisão',
    'Random Forest',
    'SVM RBF',
    'Gradient Boosting',
]


def carregar_dados():
    """Baixa o dataset UCI Phishing Websites (ID 327) e retorna X, y."""
    print('Carregando dataset...')
    dataset = fetch_ucirepo(id=327)
    X = dataset.data.features
    y = dataset.data.targets.squeeze()
    print(f'  Amostras: {len(X)}, Atributos: {X.shape[1]}')
    return X, y


def fazer_eda(X, y):
    """Imprime estatísticas básicas do dataset."""
    print('\n--- EDA ---')
    print(f'  Formato X: {X.shape}')
    print(f'  Valores faltantes: {X.isnull().sum().sum()}')
    contagens = y.value_counts()
    print(f'  Distribuição de classes:\n{contagens}')
    return {
        'n_total': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_phishing': int((y == 1).sum()),
        'n_legitimo': int((y == -1).sum()),
        'pct_phishing': float((y == 1).mean() * 100),
    }


def construir_classificadores():
    """Retorna lista de (nome, pipeline/classificador) para os 7 modelos."""
    clfs = [
        ('Regressão Logística', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=42)),
        ])),
        ('Naive Bayes', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB()),
        ])),
        ('KNN (k=5)', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ])),
        ('Árvore de Decisão', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
        )),
        ('SVM RBF', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42)),
        ])),
        ('Gradient Boosting', GradientBoostingClassifier(
            n_estimators=200, random_state=42,
        )),
    ]
    return clfs


def validar_cruzado(clfs, X_treino, y_treino):
    """Executa validação cruzada 10-fold estratificada; retorna DataFrame de F1 por fold."""
    print('\n--- Validação Cruzada 10-fold ---')
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1_macro', 'roc_auc']

    f1_por_fold = {}
    cv_resumo = {}

    for nome, clf in clfs:
        print(f'  {nome}...')
        resultado = cross_validate(
            clf, X_treino, y_treino,
            cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1,
        )
        f1_por_fold[nome] = resultado['test_f1_macro'].tolist()
        cv_resumo[nome] = {
            'acuracia_cv_media': float(resultado['test_accuracy'].mean()),
            'f1_cv_media': float(resultado['test_f1_macro'].mean()),
            'f1_cv_std': float(resultado['test_f1_macro'].std()),
            'auc_cv_media': float(resultado['test_roc_auc'].mean()),
        }

    df_f1 = pd.DataFrame(f1_por_fold)
    return df_f1, cv_resumo


def avaliar_teste(clfs, X_treino, y_treino, X_teste, y_teste):
    """Treina em todo o treino e avalia no teste; retorna dict de métricas e modelos treinados."""
    print('\n--- Avaliação no Conjunto de Teste ---')
    resultados_teste = {}
    modelos_treinados = {}

    for nome, clf in clfs:
        print(f'  {nome}...')
        t0 = time.time()
        clf.fit(X_treino, y_treino)
        tempo = time.time() - t0

        y_pred = clf.predict(X_teste)
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_teste)[:, 1]
        else:
            y_prob = clf.decision_function(X_teste)

        fpr, tpr, _ = roc_curve(y_teste, y_prob, pos_label=1)
        auc_roc = auc(fpr, tpr)

        resultados_teste[nome] = {
            'acuracia': float(accuracy_score(y_teste, y_pred)),
            'precisao': float(precision_score(y_teste, y_pred, average='macro')),
            'revocacao': float(recall_score(y_teste, y_pred, average='macro')),
            'f1': float(f1_score(y_teste, y_pred, average='macro')),
            'auc_roc': float(auc_roc),
            'tempo_treino': float(tempo),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'matriz_confusao': confusion_matrix(y_teste, y_pred).tolist(),
        }
        modelos_treinados[nome] = clf

    return resultados_teste, modelos_treinados


def teste_estatistico(df_f1):
    """Executa teste de Friedman e pós-teste de Nemenyi sobre F1 por fold."""
    print('\n--- Teste de Friedman + Nemenyi ---')
    dados = [df_f1[col].values for col in df_f1.columns]
    stat, p_valor = stats.friedmanchisquare(*dados)
    print(f'  Friedman χ²={stat:.4f}, p={p_valor:.4e}')

    nemenyi = sp.posthoc_nemenyi_friedman(df_f1.values)
    nemenyi.columns = df_f1.columns
    nemenyi.index = df_f1.columns

    return {
        'friedman_chi2': float(stat),
        'friedman_p': float(p_valor),
        'nemenyi': nemenyi.values.tolist(),
        'nomenyi_colunas': df_f1.columns.tolist(),
    }


def calcular_importancia(modelos_treinados, X):
    """Extrai importância de features do Random Forest."""
    print('\n--- Importância de Features ---')
    rf = modelos_treinados['Random Forest']
    importancias = rf.feature_importances_
    nomes_feat = X.columns.tolist()
    pares = sorted(zip(nomes_feat, importancias), key=lambda x: x[1], reverse=True)
    return pares


def gerar_figuras(resultados_teste, importancias, X_treino, y_treino, nemenyi_df):
    """Gera e salva as 4 figuras em resultados/figuras/."""
    print('\n--- Gerando Figuras ---')
    os.makedirs('resultados/figuras', exist_ok=True)

    # 1. Curvas ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    cores = plt.cm.tab10(np.linspace(0, 0.9, 7))
    for i, (nome, res) in enumerate(resultados_teste.items()):
        ax.plot(res['fpr'], res['tpr'], color=cores[i],
                label=f"{nome} (AUC={res['auc_roc']:.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC')
    ax.legend(loc='lower right', fontsize=7.5)
    plt.savefig('resultados/figuras/curvas_roc.pdf', bbox_inches='tight')
    plt.close()
    print('  curvas_roc.pdf gerado')

    # 2. Importância de features (top 15)
    top15 = importancias[:15]
    nomes_top = [p[0] for p in top15]
    vals_top = [p[1] for p in top15]
    fig, ax = plt.subplots(figsize=(5, 6))
    cores_viridis = plt.cm.viridis(np.linspace(0.2, 0.85, len(nomes_top)))
    barras = ax.barh(range(len(nomes_top)), vals_top[::-1], color=cores_viridis)
    ax.set_yticks(range(len(nomes_top)))
    ax.set_yticklabels(nomes_top[::-1], fontsize=9)
    ax.set_xlabel('Importância (Gini)')
    ax.set_title('Top 15 Atributos — Random Forest')
    plt.savefig('resultados/figuras/importancia.pdf', bbox_inches='tight')
    plt.close()
    print('  importancia.pdf gerado')

    # 3. Matrizes de confusão (grid 2×4, último subplot vazio)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    nomes_list = list(resultados_teste.keys())
    for i, ax in enumerate(axes.flat):
        if i < 7:
            nome = nomes_list[i]
            mc = np.array(resultados_teste[nome]['matriz_confusao'])
            sns.heatmap(mc, annot=True, fmt='d', ax=ax, cmap='Blues',
                        cbar=False, linewidths=0.5)
            ax.set_title(nome, fontsize=8.5)
            ax.set_xlabel('Predito', fontsize=8)
            ax.set_ylabel('Real', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('resultados/figuras/matrizes_confusao.pdf', bbox_inches='tight')
    plt.close()
    print('  matrizes_confusao.pdf gerado')

    # 4. PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_treino)
    var1, var2 = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    mask_leg = y_treino == -1
    mask_phi = y_treino == 1
    ax.scatter(X_pca[mask_leg, 0], X_pca[mask_leg, 1],
               s=5, alpha=0.4, label='Legítimo', color='steelblue')
    ax.scatter(X_pca[mask_phi, 0], X_pca[mask_phi, 1],
               s=5, alpha=0.4, label='Phishing', color='tomato')
    ax.set_xlabel(f'Componente Principal 1 ({var1:.1f}%)')
    ax.set_ylabel(f'Componente Principal 2 ({var2:.1f}%)')
    ax.set_title('PCA 2D — Conjunto de Treino')
    ax.legend(markerscale=3)
    plt.savefig('resultados/figuras/pca_2d.pdf', bbox_inches='tight')
    plt.close()
    print('  pca_2d.pdf gerado')

    return var1, var2


def exportar_macros(stats):
    """Gera resultados/macros.tex com \newcommand para cada número do artigo."""
    os.makedirs('resultados', exist_ok=True)

    eda = stats['eda']
    melhor = stats['melhor']
    friedman = stats['friedman']
    pca = stats['pca']
    importancias = stats['importancias']

    # {,} como separador decimal: funciona em modo texto e math no LaTeX
    def fmt(v):
        return f'{v:.3f}'.replace('.', '{,}')

    def fmt2(v):
        return f'{v:.2f}'.replace('.', '{,}')

    def fmt1(v):
        return f'{v:.1f}'.replace('.', '{,}')

    n_total_fmt = f'{eda["n_total"]:,}'.replace(',', '.')
    n_phi_fmt = f'{eda["n_phishing"]:,}'.replace(',', '.')
    n_leg_fmt = f'{eda["n_legitimo"]:,}'.replace(',', '.')
    pct_phi = f'{eda["pct_phishing"]:.1f}'.replace('.', '{,}') + r'\%'

    if friedman['friedman_p'] < 0.001:
        p_str = r'< 0{,}001'
    else:
        p_str = '= ' + fmt(friedman['friedman_p'])

    # percentual de importância acumulada nas duas principais features
    top2_pct = (importancias[0][1] + importancias[1][1]) * 100
    top2_str = f'{top2_pct:.1f}'.replace('.', '{,}') + r'\%'

    nb_f1 = stats['teste']['Naive Bayes']['f1']
    rf_sigma = stats['cv']['Random Forest']['f1_cv_std']

    linhas = [
        r'\newcommand{\nTotal}{' + n_total_fmt + '}',
        r'\newcommand{\nFeatures}{' + str(eda['n_features']) + '}',
        r'\newcommand{\nPhishing}{' + n_phi_fmt + '}',
        r'\newcommand{\nLegitimo}{' + n_leg_fmt + '}',
        r'\newcommand{\pctPhishing}{' + pct_phi + '}',
        r'\newcommand{\friedmanChi}{' + fmt2(friedman['friedman_chi2']) + '}',
        r'\newcommand{\friedmanP}{' + p_str + '}',
        r'\newcommand{\melhorModelo}{' + melhor['nome'] + '}',
        r'\newcommand{\melhorFOne}{' + fmt(melhor['f1']) + '}',
        r'\newcommand{\melhorAUC}{' + fmt(melhor['auc_roc']) + '}',
        r'\newcommand{\melhorSigmaCV}{' + fmt(rf_sigma) + '}',
        r'\newcommand{\naiveBayesFOne}{' + fmt(nb_f1) + '}',
        r'\newcommand{\pcaVarUm}{' + fmt1(pca['var1']) + r'\%}',
        r'\newcommand{\pcaVarDois}{' + fmt1(pca['var2']) + r'\%}',
        r'\newcommand{\topDoisPct}{' + top2_str + '}',
    ]

    with open('resultados/macros.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n')
    print('  macros.tex gerado')


def exportar_tabelas(stats, nemenyi_colunas, nemenyi_vals, importancias):
    """Gera as 3 tabelas LaTeX em resultados/."""

    def fmt_br(v, decimais=3):
        return f'{v:.{decimais}f}'.replace('.', '{,}')

    # tabela_principal.tex
    linhas_tab = []
    for nome in NOMES:
        cv = stats['cv'][nome]
        te = stats['teste'][nome]
        f1_cv = f'{fmt_br(cv["f1_cv_media"])} $\\pm$ {fmt_br(cv["f1_cv_std"])}'
        f1_te = fmt_br(te['f1'])
        auc_te = fmt_br(te['auc_roc'])
        tempo = fmt_br(te['tempo_treino'], 2)
        linhas_tab.append(f'    {nome} & {f1_cv} & {f1_te} & {auc_te} & {tempo} \\\\')

    tab_principal = (
        '\\begin{tabular}{lcccc}\n'
        '\\hline\n'
        '\\textbf{Classificador} & \\textbf{F1 (CV, $\\mu\\pm\\sigma$)} & '
        '\\textbf{F1 (teste)} & \\textbf{AUC-ROC} & \\textbf{Tempo (s)} \\\\\n'
        '\\hline\n'
        + '\n'.join(linhas_tab) + '\n'
        '\\hline\n'
        '\\end{tabular}\n'
    )
    with open('resultados/tabela_principal.tex', 'w', encoding='utf-8') as f:
        f.write(tab_principal)
    print('  tabela_principal.tex gerada')

    # tabela_nemenyi.tex
    n = len(nemenyi_colunas)
    cab = ' & '.join([f'\\textbf{{{c}}}' for c in nemenyi_colunas])
    linhas_nem = ['\\begin{tabular}{l' + 'c' * n + '}', '\\hline',
                  '\\textbf{} & ' + cab + ' \\\\', '\\hline']
    for i, linha_nome in enumerate(nemenyi_colunas):
        celulas = []
        for j in range(n):
            v = nemenyi_vals[i][j]
            if i == j:
                celulas.append('---')
            else:
                celulas.append(fmt_br(v))
        linhas_nem.append(f'    {linha_nome} & ' + ' & '.join(celulas) + ' \\\\')
    linhas_nem += ['\\hline', '\\end{tabular}']
    with open('resultados/tabela_nemenyi.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas_nem) + '\n')
    print('  tabela_nemenyi.tex gerada')

    # tabela_top_features.tex
    top10 = importancias[:10]
    linhas_feat = ['\\begin{tabular}{clc}', '\\hline',
                   '\\textbf{Pos.} & \\textbf{Atributo} & \\textbf{Importância} \\\\',
                   '\\hline']
    for i, (feat, imp) in enumerate(top10, 1):
        linhas_feat.append(f'    {i} & {feat} & {fmt_br(imp)} \\\\')
    linhas_feat += ['\\hline', '\\end{tabular}']
    with open('resultados/tabela_top_features.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas_feat) + '\n')
    print('  tabela_top_features.tex gerada')


def main():
    """Executa o pipeline completo de análise."""
    print('=== Pipeline de Detecção de Phishing ===\n')

    # 1. Carregamento
    X, y = carregar_dados()

    # 2. EDA
    eda_stats = fazer_eda(X, y)

    # 3. Split estratificado 80/20
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print(f'\nTreino: {len(X_treino)} | Teste: {len(X_teste)}')

    # 4. Classificadores
    clfs = construir_classificadores()

    # 5. Validação cruzada
    df_f1, cv_resumo = validar_cruzado(clfs, X_treino, y_treino)

    # 6. Avaliação no teste
    resultados_teste, modelos_treinados = avaliar_teste(
        clfs, X_treino, y_treino, X_teste, y_teste,
    )

    # 7. Teste estatístico
    stat_resultado = teste_estatistico(df_f1)

    # 8. Importância de features
    importancias = calcular_importancia(modelos_treinados, X)

    # 9. Melhor modelo (por F1 no teste)
    melhor_nome = max(resultados_teste, key=lambda n: resultados_teste[n]['f1'])
    melhor = {'nome': melhor_nome, **resultados_teste[melhor_nome]}

    # Estrutura unificada de stats
    nemenyi_df = pd.DataFrame(
        stat_resultado['nemenyi'],
        index=stat_resultado['nomenyi_colunas'],
        columns=stat_resultado['nomenyi_colunas'],
    )

    # 10. Figuras
    var1, var2 = gerar_figuras(
        resultados_teste, importancias, X_treino.values, y_treino.values, nemenyi_df,
    )

    stats = {
        'eda': eda_stats,
        'cv': cv_resumo,
        'teste': resultados_teste,
        'friedman': stat_resultado,
        'melhor': melhor,
        'pca': {'var1': var1, 'var2': var2},
        'importancias': importancias,
        'f1_por_fold': df_f1.to_dict(orient='list'),
    }

    # 11. Exportação
    print('\n--- Exportando resultados ---')
    exportar_macros(stats)
    exportar_tabelas(
        stats,
        stat_resultado['nomenyi_colunas'],
        stat_resultado['nemenyi'],
        importancias,
    )

    # stats.json (serialização segura)
    stats_json = {
        'eda': eda_stats,
        'cv': cv_resumo,
        'teste': {
            k: {kk: vv for kk, vv in v.items() if kk not in ('fpr', 'tpr', 'matriz_confusao')}
            for k, v in resultados_teste.items()
        },
        'friedman': {
            'chi2': stat_resultado['friedman_chi2'],
            'p': stat_resultado['friedman_p'],
        },
        'melhor': {'nome': melhor_nome, 'f1': melhor['f1'], 'auc_roc': melhor['auc_roc']},
        'pca': {'var1': var1, 'var2': var2},
        'importancias_top20': importancias[:20],
        'f1_por_fold': df_f1.to_dict(orient='list'),
    }
    with open('resultados/stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2)
    print('  stats.json gerado')

    print('\n=== Concluído com sucesso ===')
    print(f'Melhor modelo: {melhor_nome} | F1={melhor["f1"]:.4f} | AUC={melhor["auc_roc"]:.4f}')


if __name__ == '__main__':
    main()
