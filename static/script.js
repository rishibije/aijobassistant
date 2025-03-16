document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("resume")
  const fileName = document.getElementById("file-name")
  const uploadArea = document.getElementById("upload-area")

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      fileName.textContent = e.target.files[0].name
    }
  })

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault()
    uploadArea.classList.add("dragover")
  })

  uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault()
    uploadArea.classList.remove("dragover")
  })

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault()
    uploadArea.classList.remove("dragover")
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files
      fileName.textContent = e.dataTransfer.files[0].name
    }
  })
})

