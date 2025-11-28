from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - handled by requirements
    umap = None

logger = logging.getLogger(__name__)


def prepare_unsupervised_features(
    model_input: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """Prepare feature matrix for unsupervised learning with memory safeguards.

    Steps:
      1. Drop known high-cardinality identifier columns.
      2. Limit one-hot encoding to categoricals below a cardinality threshold.
      3. Impute numeric columns (median) and non-numeric ("missing").
      4. Optionally transpose if the matrix ends excessively wide vs tall.
      5. Prune near-constant columns (variance threshold) using float32.
      6. Drop extremely sparse one-hot columns (high zero ratio).
      7. Convert to sparse if enabled.
      8. Scale (StandardScaler) and return float32 DataFrame.
    """
    random_state = params.get("random_state", 42)
    max_card = params.get("max_categorical_cardinality", 50)
    variance_threshold = params.get("variance_threshold", 0.05)
    max_zero_ratio = params.get("max_zero_ratio", 0.997)
    sparsify = bool(params.get("sparsify", True))
    transpose_if_wide = bool(params.get("transpose_if_wide", True))

    df = model_input.copy()

    # 1. Drop known IDs/timestamps that explode cardinality.
    drop_cols = ["customer_id", "order_purchase_timestamp", "product_id", "seller_id"]
    to_drop_existing = [c for c in drop_cols if c in df.columns]
    if to_drop_existing:
        df = df.drop(columns=to_drop_existing)
        logger.info("Dropped identifier columns: %s", to_drop_existing)

    # 2. Controlled one-hot encoding of categoricals.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    safe_cats = []
    high_card_cats = []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= max_card:
            safe_cats.append(c)
        else:
            high_card_cats.append(c)
    if high_card_cats:
        logger.warning(
            "Skipping one-hot for high-cardinality categoricals (>%d): %s",
            max_card, high_card_cats
        )
        # Optionally we could target-frequency encode; for simplicity keep dropped now.
        df = df.drop(columns=high_card_cats)
    if safe_cats:
        df = pd.get_dummies(df, columns=safe_cats, drop_first=True)
        logger.info("One-hot encoded %d categorical columns (<=%d unique values).", len(safe_cats), max_card)

    # 3. Impute missing values.
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna("missing")

    # 4. Optional transpose safeguard (if far wider than tall).
    if transpose_if_wide and df.shape[1] > df.shape[0] * 25:  # heuristic
        logger.warning(
            "DataFrame is extremely wide (%d cols vs %d rows); transposing prior to variance filtering.",
            df.shape[1], df.shape[0]
        )
        df = df.T

    # Ensure numeric dtype for variance computation and scale down precision.
    df = df.apply(pd.to_numeric, errors="ignore")
    # Cast numeric columns to float32 to halve memory.
    for col in df.columns:
        if df[col].dtype.kind in "fbiu":
            df[col] = df[col].astype(np.float32)

    # 5. Variance threshold pruning (works on dense or sparse matrix).
    selector = VarianceThreshold(threshold=variance_threshold)
    try:
        arr = df.values  # still may be large; float32 reduces footprint.
        reduced = selector.fit_transform(arr)
    except MemoryError:
        logger.warning("MemoryError during variance threshold; falling back to chunked pruning.")
        keep_cols: list[str] = []
        chunk = 2000
        for start in range(0, df.shape[1], chunk):
            sub = df.iloc[:, start:start+chunk]
            variances = sub.var(axis=0)
            keep = variances[variances > variance_threshold].index.tolist()
            keep_cols.extend(keep)
        df = df[keep_cols]
        reduced = df.values
        selected_features = df.columns.tolist()
    else:
        selected_features = df.columns[selector.get_support()].tolist()
        df = pd.DataFrame(reduced, index=model_input.index if transpose_if_wide is False else df.index, columns=selected_features)
    logger.info("Selected %d features after variance threshold %.4f (dropped %d).",
                len(selected_features), variance_threshold, (df.shape[1] - len(selected_features)))

    # 6. Drop extremely sparse columns (mostly zeros) if still dense.
    if not df.empty:
        zero_ratio = (df == 0).sum() / len(df)
        sparse_drop = zero_ratio[zero_ratio > max_zero_ratio].index.tolist()
        if sparse_drop:
            df = df.drop(columns=sparse_drop)
            logger.info("Dropped %d ultra-sparse columns (zero_ratio>%.3f).", len(sparse_drop), max_zero_ratio)

    # 7. Convert to sparse if enabled and beneficial.
    if sparsify and df.size > 0:
        density = (df != 0).sum().sum() / df.size
        if density < 0.15:  # Only sparsify if mostly zeros
            try:
                from scipy import sparse as sp
                sp_mat = sp.csr_matrix(df.values)
                logger.info("Converted feature matrix to CSR sparse (density=%.4f).", density)
                # 8. Scale: need dense for StandardScaler; scale on limited dense reconverted.
                scaled_dense = StandardScaler().fit_transform(sp_mat.toarray()).astype(np.float32)
                features = pd.DataFrame(scaled_dense, index=df.index, columns=df.columns)
            except Exception as e:  # pragma: no cover
                logger.warning("Sparse conversion failed (%s); using dense fallback.", e)
                scaled = StandardScaler().fit_transform(df.values).astype(np.float32)
                features = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        else:
            scaled = StandardScaler().fit_transform(df.values).astype(np.float32)
            features = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    else:
        scaled = StandardScaler().fit_transform(df.values).astype(np.float32)
        features = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    # Global sampling safeguard.
    max_sample_size = params.get("max_sample_size", 10000)
    if len(features) > max_sample_size:
        features = features.sample(n=max_sample_size, random_state=random_state)
        logger.warning("Sampled down to %d rows for downstream unsupervised tasks.", max_sample_size)

    logger.info("Prepared unsupervised feature matrix with shape %s (dtype float32).", features.shape)
    return features


def _compute_cluster_metrics(X: pd.DataFrame, labels: np.ndarray, algo: str, inertia: float | None = None) -> Dict:
    unique_labels = set(labels)
    # Metrics fail when only one cluster or noise only
    if len(unique_labels - {-1}) < 2:
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski = np.nan
    else:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)

    return {
        "algorithm": algo,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski,
        "inertia": inertia,
        "n_clusters": len(unique_labels - {-1}),
    }


def run_clustering(
    features: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Train multiple clustering algorithms (KMeans, DBSCAN, Agglomerative, GMM) and collect metrics."""
    k_range: List[int] = params.get("kmeans_k_range", [3, 4, 5, 6, 7])
    dbscan_eps = params.get("dbscan_eps", 0.5)
    dbscan_min_samples = params.get("dbscan_min_samples", 5)
    agg_clusters: List[int] = params.get("agg_clusters", [3, 4, 5])
    gmm_components: List[int] = params.get("gmm_components", [3, 4, 5])
    random_state = params.get("random_state", 42)
    agglomerative_sample_size = params.get("agglomerative_sample_size", 3000)
    
    metrics = []
    elbow_rows = []

    # KMeans across range
    best_k = None
    best_silhouette = -np.inf
    kmeans_best_model = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(features)
        inertia = km.inertia_
        elbow_rows.append({"k": k, "inertia": inertia})
        score = silhouette_score(features, labels) if len(set(labels)) > 1 else -np.inf
        if score > best_silhouette:
            best_silhouette = score
            best_k = k
            kmeans_best_model = km
    if kmeans_best_model is None:
        raise ValueError("KMeans failed to fit on provided data.")

    kmeans_labels = kmeans_best_model.predict(features)
    metrics.append(_compute_cluster_metrics(features, kmeans_labels, f"kmeans_k={best_k}", kmeans_best_model.inertia_))

    # DBSCAN
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    db_labels = db.fit_predict(features)
    metrics.append(_compute_cluster_metrics(features, db_labels, "dbscan"))

    # Agglomerative (sample to avoid O(n^2) memory explosion for large n)
    if len(features) > agglomerative_sample_size:
        logger.warning(
            "Sampling from %d to %d rows for AgglomerativeClustering to avoid memory issues",
            len(features), agglomerative_sample_size,
        )
        agg_features = features.sample(n=agglomerative_sample_size, random_state=random_state)
    else:
        agg_features = features

    agg_best_labels_sample = None
    agg_best_model = None
    best_score = -np.inf
    for n_clusters in agg_clusters:
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_sample = agg.fit_predict(agg_features)
        score = silhouette_score(agg_features, labels_sample) if len(set(labels_sample)) > 1 else -np.inf
        if score > best_score:
            best_score = score
            agg_best_labels_sample = labels_sample
            agg_best_model = agg
    if agg_best_model is None or agg_best_labels_sample is None:
        logger.warning("Agglomerative clustering skipped or failed; metrics set to NaN.")
        metrics.append(_compute_cluster_metrics(agg_features, np.full(len(agg_features), -1), "agglomerative_skipped", None))
        # Full label vector filled with -1
        agg_best_labels_full = np.full(len(features), -1, dtype=int)
    else:
        # Propagate sample labels to full index (others set to -1)
        agg_best_labels_full = np.full(len(features), -1, dtype=int)
        sample_index = agg_features.index
        # Map sample index positions
        sample_pos = features.index.get_indexer(sample_index)
        agg_best_labels_full[sample_pos] = agg_best_labels_sample
        metrics.append(_compute_cluster_metrics(agg_features, agg_best_labels_sample, f"agglomerative_k={agg_best_model.n_clusters}", None))

    # Gaussian Mixture Model (GMM)
    gmm_best_labels = None
    gmm_best_model = None
    best_score = -np.inf
    for n_components in gmm_components:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
        gmm.fit(features)
        labels = gmm.predict(features)
        score = silhouette_score(features, labels) if len(set(labels)) > 1 else -np.inf
        if score > best_score:
            best_score = score
            gmm_best_labels = labels
            gmm_best_model = gmm
    if gmm_best_model is None or gmm_best_labels is None:
        raise ValueError("GMM clustering failed.")
    metrics.append(_compute_cluster_metrics(features, gmm_best_labels, f"gmm_k={gmm_best_model.n_components}", None))

    clustering_models = {
        "kmeans": kmeans_best_model,
        "dbscan": db,
        "agglomerative": agg_best_model,
        "gmm": gmm_best_model,
    }

    cluster_labels = pd.DataFrame(
        {
            "kmeans_label": kmeans_labels,
            "dbscan_label": db_labels,
            "agglomerative_label": agg_best_labels_full,
            "gmm_label": gmm_best_labels,
        },
        index=features.index,
    )

    metrics_df = pd.DataFrame(metrics)
    elbow_df = pd.DataFrame(elbow_rows)

    logger.info("Clustering completed: best_k=%s, metrics rows=%d", best_k, len(metrics_df))
    return cluster_labels, metrics_df, elbow_df, clustering_models


def reduce_dimensionality(
    features: pd.DataFrame, cluster_labels: pd.DataFrame, params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute PCA + t-SNE + UMAP embeddings and PCA details (explained variance, loadings, contributions)."""
    random_state = params.get("random_state", 42)
    tsne_perplexity = params.get("tsne_perplexity", 30)
    n_components = params.get("dim_components", 2)

    pca = PCA(n_components=n_components, random_state=random_state)
    pca_emb = pca.fit_transform(features)
    pca_df = pd.DataFrame(
        pca_emb, columns=[f"pca_{i+1}" for i in range(pca_emb.shape[1])], index=features.index
    )
    pca_df["kmeans_label"] = cluster_labels["kmeans_label"]

    explained_var = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    # PCA loadings (feature contributions per component)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features.columns,
        columns=[f"PC{i+1}" for i in range(len(pca.components_))],
    ).reset_index().rename(columns={"index": "feature"})

    # PCA contributions: top positive/negative per component (configurable via params)
    top_n = int(params.get("pca_top_n_features", 10))
    contrib_rows: list[dict] = []
    for comp in [c for c in loadings.columns if c.startswith("PC")]:
        # Positive
        top_pos = loadings.nlargest(top_n, comp)[["feature", comp]].copy()
        for _, r in top_pos.iterrows():
            contrib_rows.append({
                "component": comp,
                "direction": "positive",
                "feature": r["feature"],
                "loading": float(r[comp]),
            })
        # Negative
        top_neg = loadings.nsmallest(top_n, comp)[["feature", comp]].copy()
        for _, r in top_neg.iterrows():
            contrib_rows.append({
                "component": comp,
                "direction": "negative",
                "feature": r["feature"],
                "loading": float(r[comp]),
            })
    pca_contributions = pd.DataFrame(contrib_rows)

    tsne = TSNE(
        n_components=2,
        perplexity=min(tsne_perplexity, max(5, len(features) // 3)),
        random_state=random_state,
        init="pca",
    )
    tsne_emb = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_emb, columns=["tsne_1", "tsne_2"], index=features.index)
    tsne_df["kmeans_label"] = cluster_labels["kmeans_label"]

    if umap is None:
        logger.warning("umap-learn not installed; skipping UMAP embedding")
        umap_df = pd.DataFrame(index=features.index, columns=["umap_1", "umap_2", "kmeans_label"])
    else:
        reducer = umap.UMAP(
            n_neighbors=params.get("umap_n_neighbors", 15),
            min_dist=params.get("umap_min_dist", 0.1),
            n_components=2,
            random_state=random_state,
        )
        umap_emb = reducer.fit_transform(features)
        umap_df = pd.DataFrame(umap_emb, columns=["umap_1", "umap_2"], index=features.index)
        umap_df["kmeans_label"] = cluster_labels["kmeans_label"]

    return pca_df, tsne_df, umap_df, explained_var, loadings, pca_contributions


def detect_anomalies(features: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Run Isolation Forest and LOF to flag potential anomalies."""
    random_state = params.get("random_state", 42)
    contamination = params.get("anomaly_contamination", 0.05)
    
    # Isolation Forest
    iso = IsolationForest(
        random_state=random_state,
        contamination=contamination,
        n_estimators=params.get("anomaly_estimators", 200),
        n_jobs=-1,
    )
    iso.fit(features)
    iso_scores = iso.decision_function(features)
    iso_preds = iso.predict(features)  # -1 = anomaly
    
    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(
        n_neighbors=params.get("lof_n_neighbors", 20),
        contamination=contamination,
        n_jobs=-1
    )
    lof_preds = lof.fit_predict(features)
    lof_scores = lof.negative_outlier_factor_
    
    # Combinar resultados
    return pd.DataFrame(
        {
            "anomaly_score_iso": iso_scores,
            "is_anomaly_iso": iso_preds == -1,
            "anomaly_score_lof": lof_scores,
            "is_anomaly_lof": lof_preds == -1,
            "is_anomaly_consensus": (iso_preds == -1) | (lof_preds == -1),
        },
        index=features.index,
    )


def profile_clusters(
    model_input: pd.DataFrame, cluster_labels: pd.DataFrame
) -> pd.DataFrame:
    """Generate statistical profiles for each cluster.
    
    Args:
        model_input: Original dataset with all features
        cluster_labels: DataFrame with cluster assignments
    
    Returns:
        DataFrame with statistical summaries per cluster
    """
    # Combinar datos con labels
    data_with_clusters = model_input.copy()
    data_with_clusters['cluster'] = cluster_labels['kmeans_label'].values
    
    # Seleccionar columnas numéricas
    numeric_cols = data_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    
    # Calcular estadísticas por cluster
    cluster_profiles = []
    for cluster_id in sorted(data_with_clusters['cluster'].unique()):
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': 100 * len(cluster_data) / len(data_with_clusters)
        }
        
        # Agregar promedios de features clave (primeras 10)
        for col in numeric_cols[:10]:
            profile[f'{col}_mean'] = cluster_data[col].mean()
            profile[f'{col}_std'] = cluster_data[col].std()
        
        cluster_profiles.append(profile)
    
    profiles_df = pd.DataFrame(cluster_profiles)
    logger.info("Generated profiles for %d clusters", len(profiles_df))
    return profiles_df


def build_cluster_enriched_dataset(
    model_input: pd.DataFrame, cluster_labels: pd.DataFrame
) -> pd.DataFrame:
    """Append cluster labels to the original model_input for supervised integration."""
    aligned = cluster_labels.reindex(model_input.index)
    enriched = model_input.copy()
    for col in aligned.columns:
        enriched[col] = aligned[col]
    return enriched
