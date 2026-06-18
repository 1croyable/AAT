# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="./checkerboard3d_l1_child_sweep/results.csv",
        help="Path to results.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df = to_num(
        df,
        [
            "fixed_k",
            "selected_k",
            "max_children",
            "params",
            "best_epoch",
            "best_val_acc",
            "best_val_f1",
            "test_acc",
            "test_f1",
            "final_val_acc",
            "train_time_sec",
            "val_fisher_after",
            "diag_gate_active",
            "diag_move_norm",
        ],
    )

    fixed = df[df["mode"] == "fixed"].copy().sort_values("fixed_k")
    auto = df[df["mode"] == "auto"].copy().sort_values("max_children")

    if fixed.empty:
        raise RuntimeError("No fixed-K rows found in CSV.")

    print("\n=== Basic info ===")
    print(f"csv: {csv_path}")
    print(f"fixed runs: {len(fixed)}")
    print(f"auto runs : {len(auto)}")
    print(f"K range   : {int(fixed['fixed_k'].min())} -> {int(fixed['fixed_k'].max())}")

    best_val_row = fixed.loc[fixed["best_val_acc"].idxmax()]
    best_test_row = fixed.loc[fixed["test_acc"].idxmax()]
    first_95 = fixed[fixed["best_val_acc"] >= 0.95]

    print("\n=== Best by validation accuracy ===")
    print(best_val_row[
        [
            "run_id", "fixed_k", "best_val_acc", "test_acc",
            "final_val_acc", "val_fisher_after", "diag_gate_active", "diag_move_norm"
        ]
    ].to_string())

    print("\n=== Best by test accuracy ===")
    print(best_test_row[
        [
            "run_id", "fixed_k", "best_val_acc", "test_acc",
            "final_val_acc", "val_fisher_after", "diag_gate_active", "diag_move_norm"
        ]
    ].to_string())

    if not first_95.empty:
        row95 = first_95.iloc[0]
        print(f"\nFirst K reaching val_acc >= 0.95: K={int(row95['fixed_k'])}, val={row95['best_val_acc']:.6f}")

    print("\n=== Top 10 by best_val_acc ===")
    print(
        fixed.sort_values("best_val_acc", ascending=False)[
            [
                "fixed_k", "best_val_acc", "test_acc", "final_val_acc",
                "val_fisher_after", "diag_gate_active", "diag_move_norm"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== Top 10 by test_acc ===")
    print(
        fixed.sort_values("test_acc", ascending=False)[
            [
                "fixed_k", "best_val_acc", "test_acc", "final_val_acc",
                "val_fisher_after", "diag_gate_active", "diag_move_norm"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # -------------------------------
    # Figure 1: full performance curve
    # -------------------------------
    plt.figure(figsize=(13, 6))
    plt.plot(fixed["fixed_k"], fixed["best_val_acc"], marker="o", markersize=3, label="fixed best_val_acc")
    plt.plot(fixed["fixed_k"], fixed["test_acc"], marker="o", markersize=3, label="fixed test_acc")
    plt.plot(fixed["fixed_k"], fixed["final_val_acc"], marker="o", markersize=3, label="fixed final_val_acc")

    if not auto.empty:
        plt.scatter(auto["selected_k"], auto["best_val_acc"], s=140, marker="*", label="auto best_val_acc")
        plt.scatter(auto["selected_k"], auto["test_acc"], s=140, marker="X", label="auto test_acc")
        for _, r in auto.iterrows():
            x = r["selected_k"]
            y = r["best_val_acc"]
            label = f"auto max{int(r['max_children'])}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.xlabel("K (children per class)")
    plt.ylabel("Accuracy")
    plt.title("Checkerboard3D L1 child sweep - full performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -------------------------------
    # Figure 2: zoomed top region
    # -------------------------------
    top_region = fixed[fixed["fixed_k"] >= 25].copy()
    if not top_region.empty:
        plt.figure(figsize=(13, 6))
        plt.plot(top_region["fixed_k"], top_region["best_val_acc"], marker="o", markersize=4, label="fixed best_val_acc")
        plt.plot(top_region["fixed_k"], top_region["test_acc"], marker="o", markersize=4, label="fixed test_acc")
        plt.plot(top_region["fixed_k"], top_region["final_val_acc"], marker="o", markersize=4, label="fixed final_val_acc")

        if not auto.empty:
            auto_in = auto[(auto["selected_k"] >= 25) & (auto["selected_k"] <= top_region["fixed_k"].max())]
            if not auto_in.empty:
                plt.scatter(auto_in["selected_k"], auto_in["best_val_acc"], s=160, marker="*", label="auto best_val_acc")
                plt.scatter(auto_in["selected_k"], auto_in["test_acc"], s=160, marker="X", label="auto test_acc")
                for _, r in auto_in.iterrows():
                    x = r["selected_k"]
                    y = r["best_val_acc"]
                    label = f"auto max{int(r['max_children'])}"
                    plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

        ymin = top_region[["best_val_acc", "test_acc", "final_val_acc"]].min().min() - 0.01
        ymax = top_region[["best_val_acc", "test_acc", "final_val_acc"]].max().max() + 0.005
        plt.ylim(ymin, ymax)

        plt.xlabel("K (children per class)")
        plt.ylabel("Accuracy")
        plt.title("Checkerboard3D L1 child sweep - zoomed top region (K >= 25)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # -------------------------------
    # Figure 3: fisher after transport
    # -------------------------------
    plt.figure(figsize=(13, 5))
    plt.plot(fixed["fixed_k"], fixed["val_fisher_after"], marker="o", markersize=3, label="val_fisher_after")
    if not auto.empty:
        plt.scatter(auto["selected_k"], auto["val_fisher_after"], s=140, marker="*", label="auto val_fisher_after")
        for _, r in auto.iterrows():
            x = r["selected_k"]
            y = r["val_fisher_after"]
            label = f"auto max{int(r['max_children'])}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    plt.xlabel("K (children per class)")
    plt.ylabel("Fisher score")
    plt.title("Checkerboard3D L1 child sweep - val_fisher_after")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # -------------------------------
    # Figure 4: diagnostics
    # -------------------------------
    plt.figure(figsize=(13, 5))
    plt.plot(fixed["fixed_k"], fixed["diag_gate_active"], marker="o", markersize=3, label="diag_gate_active")
    plt.plot(fixed["fixed_k"], fixed["diag_move_norm"], marker="o", markersize=3, label="diag_move_norm")
    if not auto.empty:
        plt.scatter(auto["selected_k"], auto["diag_gate_active"], s=140, marker="*", label="auto gate_active")
        plt.scatter(auto["selected_k"], auto["diag_move_norm"], s=140, marker="X", label="auto move_norm")

    plt.xlabel("K (children per class)")
    plt.ylabel("Diagnostic value")
    plt.title("Checkerboard3D L1 child sweep - diagnostics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()