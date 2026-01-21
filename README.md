# Rag_adaptive_vllm

## 🔧 vLLM Fork 与 Submodule 使用说明（重要）

本项目依赖 **vLLM**，并且在研究与实验过程中 **对 vLLM 源码进行了定制修改**。
为避免 **误将修改推送到 vLLM 官方仓库**，同时保证 **代码管理规范性与实验可复现性**，本项目采用 **Fork + Submodule** 的方式管理 vLLM。

---

### 1️⃣ 使用 Fork 的 vLLM 仓库（而非官方仓库）

* **vLLM 官方仓库**：
    `https://github.com/vllm-project/vllm`

* **本项目使用的 vLLM Fork 仓库（可安全推送修改）**：
    `https://github.com/Moolitan/vllm`

> **注意**：所有对 vLLM 的修改 **仅提交到 Fork 仓库**，不会推送到官方仓库。

---

### 2️⃣ vLLM 以 Git Submodule 的形式集成

在主仓库 `Rag_adaptive_vllm` 中，`vllm/` 目录被配置为 Git submodule，指向上述 Fork 仓库：

```text
Rag_adaptive_vllm/
├── vllm/        # Git submodule → [https://github.com/Moolitan/vllm]
├── adaptive_vllm.py
├── README.md
└── ...
```

主仓库 仅记录 vLLM 对应的 commit 指针，而不直接包含 vLLM 的完整源码历史。

### 3️⃣ 克隆本项目（包含 vLLM Submodule）

* **推荐使用以下方式一次性克隆完整项目**：

    `git clone --recurse-submodules https://github.com/Moolitan/Rag_adaptive_vllm.git`


* **如果已先克隆主仓库，可通过以下命令补充拉取 submodule**：

    `git submodule update --init --recursive`

### 4️⃣ 修改 vLLM 的正确工作流（⚠️ 非常重要）
#### Step A：在 vLLM 子模块中修改并提交（推送到 Fork）
     cd vllm
     
修改 vLLM 源码
```text
git add -A
git commit -m "feat: customized vLLM for adaptive RAG"
git push
```

⚠️ vLLM 子模块的 origin 指向 Fork 仓库 `https://github.com/Moolitan/vllm`

vLLM 官方仓库仅作为 upstream 使用，且 已禁用 push 操作。

#### Step B：在主仓库中更新 submodule 指针
```text
cd ..
git add vllm
git commit -m "Update vLLM submodule to latest fork commit"
git push
```

主仓库仅记录 当前项目所使用的 vLLM 版本（commit），从而保证实验与结果的可复现性。
