# Deploying Weakly Supervised Segmentation Studio on Cloudflare

Because this application relies on **PyTorch**, **Torchvision**, and **Segmentation Models PyTorch**, it requires a Python execution runtime to perform neural network inference. Cloudflare Pages/Workers run on JavaScript V8 edge nodes, so deploying Python ML apps with Cloudflare is typically accomplished via **Cloudflare Tunnel (`cloudflared`)** or **Cloudflare Edge Proxy + Python Cloud Host**.

---

## Option 1: Cloudflare Tunnel (Recommended for Direct Cloudflare Deployment)

[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) connects your local or cloud Python server directly to Cloudflare's Global Edge Network without exposing public ports or modifying firewall rules.

### Quick Start (Instant Live HTTPS URL)
1. Start the Gradio application locally:
   ```bash
   python app.py
   ```
2. Open a second terminal and run Cloudflare Quick Tunnel:
   ```bash
   npx cloudflared tunnel --url http://127.0.0.1:7860
   ```
3. Cloudflare will output a public HTTPS URL (e.g., `https://random-subdomain.trycloudflare.com`).

### Production Deployment (Custom Cloudflare Domain)
1. Install `cloudflared`:
   - **Windows:** `winget install Cloudflare.cloudflared`
   - **Linux:** `sudo apt-get install cloudflared`
2. Authenticate with Cloudflare:
   ```bash
   cloudflared tunnel login
   ```
3. Create a persistent tunnel:
   ```bash
   cloudflared tunnel create weakly-seg-app
   ```
4. Configure routing to your custom domain:
   ```bash
   cloudflared tunnel route dns weakly-seg-app segmentation.yourdomain.com
   ```
5. Run the tunnel:
   ```bash
   cloudflared tunnel run weakly-seg-app --url http://127.0.0.1:7860
   ```

---

## Option 2: Cloudflare Pages + Hosted ML Engine (HuggingFace / Docker)

If you prefer to host the PyTorch inference backend on cloud infrastructure (e.g. HuggingFace Spaces, Railway, or AWS) while serving the frontend via Cloudflare Edge:

1. **Deploy Python Backend to Hugging Face Spaces / Modal / Railway:**
   - Push repository to Hugging Face Spaces (Gradio SDK template).
   - Get the live Space URL: `https://username-weak-segmentation.hf.space`.

2. **Deploy Gateway Page to Cloudflare Pages:**
   - Create `public/index.html` with full-screen iframe pointing to your hosted ML app:
     ```html
     <!DOCTYPE html>
     <html lang="en">
     <head>
       <meta charset="UTF-8">
       <title>Weakly Supervised Segmentation Studio</title>
       <style>html, body, iframe { width: 100%; height: 100%; margin: 0; padding: 0; border: none; }</style>
     </head>
     <body>
       <iframe src="https://username-weak-segmentation.hf.space"></iframe>
     </body>
     </html>
     ```
   - Deploy to Cloudflare Pages using Wrangler CLI:
     ```bash
     npx wrangler pages deploy public --project-name weakly-segmentation
     ```

---

## Technical Specifications
- **Local Application URL:** `http://127.0.0.1:7860`
- **Framework:** Gradio 6.0 + PyTorch U-Net
- **Dataset Previews:** Embedded light subset (`assets/preview_images` and `assets/preview_masks`) - 0 runtime download required.
