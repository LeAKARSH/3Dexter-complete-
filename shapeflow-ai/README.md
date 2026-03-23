<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/2b3da288-da58-489b-9b38-f97cb3b45cfe

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Start the local Shape-E Python service from `Shape_E.py.py`
3. Set the service URLs in [.env.local](.env.local)
4. Run the app:
   `npm run dev`

## Pipeline Notes

- Organic mode is handled by the local Python Shape-E service.
- Parametric mode is intentionally left as an integration point for your custom fine-tuned model.
- To plug in the parametric model later, expose an HTTP endpoint and set `PARAMETRIC_MODEL_URL`.
