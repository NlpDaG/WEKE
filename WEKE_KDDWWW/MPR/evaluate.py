from ke_edge_features import main


if __name__ == "__main__":
    datasets = ["WWW", "KDD"]
    parts = 'WEKEsdc+node'
    # sub_vec_types = ['c.emb', 't.emb', 'c+t.emb', 'h.emb']
    sub_vec_types = ['c.emb']
    damping = 0.85

    for ds in datasets:
        for svt in sub_vec_types:
            main(ds, parts, svt, damping)
