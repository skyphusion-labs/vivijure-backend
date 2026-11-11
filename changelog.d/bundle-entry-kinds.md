**Fix: a bundle extract accepts only regular files and directories.**

Absolute names and names with a `..` component are refused. A link is refused, including one whose target resolves outside the destination. A fifo, a character device, and a block device are refused.
