

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
