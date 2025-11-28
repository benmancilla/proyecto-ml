from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    prepare_unsupervised_features,
    run_clustering,
    reduce_dimensionality,
    detect_anomalies,
    profile_clusters,
    build_cluster_enriched_dataset,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_unsupervised_features,
                inputs=["model_input", "params:unsupervised"],
                outputs="unsupervised_features",
                name="prepare_unsupervised_features_node",
            ),
            node(
                func=run_clustering,
                inputs=["unsupervised_features", "params:unsupervised"],
                outputs=[
                    "cluster_labels",
                    "clustering_metrics",
                    "clustering_elbow",
                    "clustering_models",
                ],
                name="run_clustering_node",
            ),
            node(
                func=reduce_dimensionality,
                inputs=["unsupervised_features", "cluster_labels", "params:unsupervised"],
                outputs=[
                    "pca_embedding",
                    "tsne_embedding",
                    "umap_embedding",
                    "pca_explained_variance",
                    "pca_loadings",
                    "pca_contributions",
                ],
                name="reduce_dimensionality_node",
            ),
            node(
                func=detect_anomalies,
                inputs=["unsupervised_features", "params:unsupervised"],
                outputs="anomaly_scores",
                name="detect_anomalies_node",
            ),
            node(
                func=profile_clusters,
                inputs=["unsupervised_features", "cluster_labels"],
                outputs="cluster_profiles",
                name="profile_clusters_node",
            ),
            node(
                func=build_cluster_enriched_dataset,
                inputs=["unsupervised_features", "cluster_labels"],
                outputs="model_input_with_clusters",
                name="build_cluster_enriched_dataset_node",
            ),
        ]
    )
