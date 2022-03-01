This directory contains the git patches for PyTorch. CATCH strongly depends on these patches to work,
so before compiling PyTorch, these patches should be applied to PyTorch. For how to apply these patches,
you can refer the CONTRIBUTING.md file in catch.

If you want to make patches for PyTorch, here are some tips:

* Change the files that you need in PyTorch.

* Use `git diff` command to generate the patch.

For example, after you have changed the PyTorch files, you can use the following command:

```bash
git diff > your_generated_patch.diff
```

In addition, please notice that the associated changes are suggested arranging into one patch file.

* Place your generated patch into catch/pytorch_patches folder
