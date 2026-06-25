import os
import subprocess
import sys

# Konfigurasi repositori Dagshub & GitHub
REPO_OWNER = "Npppss"
REPO_NAME = "Hands-On-Sistem-Rekomendasi-Film"
DAGSHUB_URL = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}"

print("=== MLOps Setup: DVC & Dagshub ===")
print(f"Target Repository: {DAGSHUB_URL}\n")

def run_command(command, description):
    print(f"Running: {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during '{description}':\n{e.stderr.strip()}")
        return False

# 1. Pengecekan Git lokal
git_active = os.path.exists(".git") or run_command("git status", "Check Git Status")
if not git_active:
    print("Warning: Repositori Git lokal tidak terdeteksi.")
    print("Menginisialisasi DVC dengan mode '--no-scm' (tanpa Git control) sebagai fallback...")
    dvc_init_cmd = "dvc init --no-scm --force"
else:
    print("Git terdeteksi aktif. Menginisialisasi DVC standard...")
    dvc_init_cmd = "dvc init --force"

# 2. Inisialisasi DVC
if not run_command(dvc_init_cmd, "Initialize DVC"):
    print("Gagal menginisialisasi DVC. Pastikan DVC sudah terinstal di environment Anda.")
    sys.exit(1)

# 3. Tambahkan Remote Storage Dagshub di DVC
dvc_remote_url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.dvc"
run_command(f"dvc remote add -f -d origin {dvc_remote_url}", "Add Dagshub DVC Remote")

# Modifikasi remote auth settings
run_command("dvc remote modify origin auth basic", "Configure Remote Auth Method")
run_command(f"dvc remote modify origin user {REPO_OWNER}", "Configure Remote Username")
run_command("dvc remote modify origin ask_password true", "Set DVC to Ask Password/Token when Pushing")

# 4. Tambahkan folder dataset ke dalam DVC tracking
dataset_folder = "data"
if os.path.exists(dataset_folder):
    run_command(f"dvc add {dataset_folder}", f"Add dataset folder '{dataset_folder}' to DVC")
    print(f"\n[SUKSES] Folder '{dataset_folder}' sekarang dilacak oleh DVC.")
    print(f"File '{dataset_folder}.dvc' dan konfigurasi '.dvc/config' telah dibuat.")
    print("Data asli otomatis diabaikan oleh Git via .gitignore.")
else:
    print(f"\nWarning: Folder dataset '{dataset_folder}' tidak ditemukan. Silakan jalankan 'dvc add <path_data>' secara manual nanti.")

print("\n=== SETUP SELESAI ===")
print("Langkah selanjutnya untuk mengunggah dataset Anda ke Dagshub Remote Storage:")
print("1. Jalankan perintah berikut untuk mengunggah data (masukkan token akses Dagshub Anda sebagai password jika diminta):")
print("   dvc push -r origin")
print("2. Commit perubahan metadata DVC Anda ke Git:")
print("   git add .dvc/config data.dvc .gitignore")
print("   git commit -m \"Add DVC tracking for movies dataset\"")
print("   git push origin main")
