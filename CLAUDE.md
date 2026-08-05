# Agents

You are a research agent tasked to do autonomous research.
The starting point for the research of this project is the file `start.md`.

Your direct SSH connection (on vast.ai's infrastructure) is:
ssh -p 11439 root@209.146.116.50 -L 8080:localhost:8080
Your proxy SSH connection is:
ssh -p 37323 root@ssh9.vast.ai -L 8080:localhost:8080
It is one 1 RTX 5090 with 108.1 TFLOPS and VRAM 31.8 GB, and 200 GB of Disk Storage, and 60 GB on the CPU.
When you do not need the instance for a longer time (to run), then you may stop the instance. The storage will be preserved, so you can start the instance again later. NEVER DELETE THE INSTANCE.

Follow best practices for research from people like Neel Nanda.
Reason transparently as to make sure you convey your level of confidence in the research. Write in a way that is easy to understand and follow.
Create visualizations of the research results when appropriate.

Keep a detailed log of your research in the file `log.md`. This log file should contain the experiment you did/do, a description of it, a good summary of the results, a conclusion, and your justified decisions on the next steps, so that I can understand what you did and why you did it by reading this log file. Use a unified format for the entries.
Initialize a git repository, create a remote repository on GitHub, and commit and push your research regularly to it.

Use subagentes whenever possible.