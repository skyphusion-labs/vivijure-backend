**Fix: `R2.exists()` returns False only for a real not-found (#460).**

It caught every `ClientError` and returned False, so an authorization or expired-credential error
read as "object absent". Now only `404`, `NoSuchKey` and `NotFound` return False; any other error
propagates out of `exists()`. Classification is duck-typed on `ClientError.response`, so no botocore
import is needed.
