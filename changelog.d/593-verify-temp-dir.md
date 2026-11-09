### fix(verify): a verify render removes its temp directory

`verify._pod_draft_render` downloaded the finished clip into a `vj-verify-*` directory from `tempfile.mkdtemp` and never removed it, so every verify run left its clip on the pod's disk and every test run left one in `$TMPDIR` (over 100 found in one account). The directory is now removed in a `finally`, after a normal return and after a failed download alike.
