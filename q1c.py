def run():
    print """Compare and contrast both strategies: What are potential pitfalls with each?

    OneVsAll is computationally efficient (only `n_classes`
    classifiers are needed). One advantage of this approach is its
    interpretability. Since each class is represented by only one classifier,
    it is possible to gain knowledge about the class by inspecting its
    corresponding classifier.

    OneVsOne requires to fit each combination of classifiers
    (`n_classes * (n_classes - 1) / 2`). This method is usually slower
    than OneVsAll. But this method may be advantageous for some
    algorithms such as kernel algorithms that don't scale well with
    `n_samples`. This is because each individual learning problem only involves
    a small subset of the data. In OneVsAll classification the complete
    dataset is used `n_classes` times.
    """
