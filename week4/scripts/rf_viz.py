# 随机森林、树
import tempfile
import shutil
from pathlib import Path
from sklearn.tree import export_graphviz
import graphviz


def visualize_rf_tree(model, X, tree_idx: int = 0, max_depth: int = 3):
    estimator = model.estimators_[tree_idx]
    dot_data = export_graphviz(
        estimator,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth,
        proportion=True,
        precision=2
    )
    graph = graphviz.Source(dot_data)

    save_dir = Path(__file__).resolve().parent.parent / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("png", "pdf"):
            graph.render(
                filename=f"rf_tree_{tree_idx}",
                directory=tmp,
                format=fmt,
                cleanup=True
            )
            src = Path(tmp) / f"rf_tree_{tree_idx}.{fmt}"
            dst = save_dir / f"rf_tree_{tree_idx}.{fmt}"
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)

    print(f"[RandomForest] 第 {tree_idx} 棵树已保存到 {save_dir}/rf_tree_{tree_idx}.png/.pdf")