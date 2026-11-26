from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import base64
from io import BytesIO

app = FastAPI(title="Acrobot PPO Comparison")

# ====================== تحميل الموديلين ======================
print("🔄 جاري تحميل الموديلين...")
pre_model = PPO.load("ppo-Acrobot-v1.zip", device="cpu")
ft_model  = PPO.load("best_acrobot_model.zip", device="cpu")
print("✅ تم تحميل الموديلين بنجاح!\n")

def evaluate(model, n=20):
    env = gym.make("Acrobot-v1")
    mean, std = evaluate_policy(model, env, n_eval_episodes=n, deterministic=True)
    env.close()
    return round(float(mean), 2), round(float(std), 2)

# ====================== توليد GIF للروبوت ======================
def generate_play_gif(model, max_steps=300, fps=30):
    env = gym.make("Acrobot-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            break

    env.close()

    # تحويل إلى GIF
    gif_buffer = BytesIO()
    imageio.mimsave(gif_buffer, frames, format='GIF', fps=fps, loop=0)
    gif_buffer.seek(0)
    
    return base64.b64encode(gif_buffer.read()).decode('utf-8')

# ====================== الصفحة الرئيسية (HTML) ======================
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Acrobot PPO - قبل وبعد</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-950 text-white min-h-screen">
      <div class="max-w-5xl mx-auto p-8">
        <h1 class="text-5xl font-bold text-center mb-2">🤖 Acrobot-v1</h1>
        <p class="text-center text-green-400 text-2xl mb-10">PPO قبل وبعد Fine-Tuning</p>

        <div class="flex justify-center gap-6 mb-12">
          <button onclick="runComparison()" 
                  class="bg-green-600 hover:bg-green-700 px-10 py-5 rounded-2xl text-2xl font-bold flex items-center gap-4 shadow-lg">
            <i class="fas fa-bolt"></i>
            <span>ابدأ المقارنة</span>
          </button>

          <button onclick="playRobot()" 
                  class="bg-blue-600 hover:bg-blue-700 px-10 py-5 rounded-2xl text-2xl font-bold flex items-center gap-4 shadow-lg">
            <i class="fas fa-play"></i>
            <span>شاهد الروبوت يلعب</span>
          </button>
        </div>

        <!-- Loading -->
        <div id="loading" class="hidden text-center py-12">
          <i class="fas fa-spinner fa-spin text-6xl mb-6 text-green-400"></i>
          <p class="text-xl">جاري التقييم...</p>
        </div>

        <!-- Results -->
        <div id="results" class="hidden grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div class="bg-gray-900 p-8 rounded-3xl border border-blue-500">
            <h2 class="text-blue-400 text-xl mb-3">الموديل الأصلي</h2>
            <div id="pre-mean" class="text-6xl font-bold mb-2"></div>
            <p class="text-gray-400">متوسط المكافأة</p>
          </div>

          <div class="bg-gray-900 p-8 rounded-3xl border border-green-500 relative">
            <div class="absolute -top-4 right-6 bg-green-600 text-sm px-4 py-1 rounded-full font-bold">محسن</div>
            <h2 class="text-green-400 text-xl mb-3">بعد Fine-Tuning</h2>
            <div id="ft-mean" class="text-6xl font-bold mb-2"></div>
            <p class="text-gray-400">متوسط المكافأة</p>
          </div>

          <div class="bg-gray-900 p-8 rounded-3xl border border-yellow-400">
            <h2 class="text-yellow-400 text-xl mb-3">الفرق</h2>
            <div id="delta" class="text-6xl font-bold mb-2"></div>
            <p id="delta-text" class="text-xl"></p>
          </div>
        </div>

        <!-- Video / GIF Section -->
        <div id="video-section" class="hidden bg-gray-900 p-8 rounded-3xl">
          <h2 class="text-3xl font-bold mb-6 text-center">🎥 الروبوت بعد Fine-Tuning</h2>
          <div class="flex justify-center">
            <img id="robot-gif" class="rounded-2xl shadow-2xl max-w-full border border-gray-700" alt="Acrobot Playing">
          </div>
          <p class="text-center text-gray-400 mt-4">يتم توليد الـ GIF الآن...</p>
        </div>
      </div>

      <script>
        async function runComparison() {
          document.getElementById('loading').classList.remove('hidden');
          document.getElementById('results').classList.add('hidden');
          document.getElementById('video-section').classList.add('hidden');

          const res = await fetch('/compare?n_eval=30');
          const data = await res.json();

          document.getElementById('pre-mean').textContent = data.pretrained.mean;
          document.getElementById('ft-mean').textContent = data.finetuned.mean;
          
          const deltaEl = document.getElementById('delta');
          deltaEl.textContent = data.delta >= 0 ? `+${data.delta}` : data.delta;
          deltaEl.className = data.delta >= 0 ? "text-6xl font-bold mb-2 text-green-400" : "text-6xl font-bold mb-2 text-red-400";

          document.getElementById('delta-text').innerHTML = 
            data.improved ? "✅ تحسن واضح" : "⚠️ لم يحدث تحسن";

          document.getElementById('loading').classList.add('hidden');
          document.getElementById('results').classList.remove('hidden');
        }

        async function playRobot() {
          document.getElementById('video-section').classList.remove('hidden');
          const gifImg = document.getElementById('robot-gif');
          gifImg.src = ""; // reset

          const res = await fetch('/play');
          const data = await res.json();
          
          gifImg.src = "data:image/gif;base64," + data.gif;
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/compare")
async def compare(n_eval: int = 20):
    pre_mean, pre_std = evaluate(pre_model, n_eval)
    ft_mean, ft_std   = evaluate(ft_model, n_eval)
    delta = round(ft_mean - pre_mean, 2)

    return {
        "pretrained": {"mean": pre_mean, "std": pre_std},
        "finetuned":  {"mean": ft_mean,  "std": ft_std},
        "delta": delta,
        "improved": delta > 0
    }

@app.get("/play")
async def play():
    """يعرض الروبوت بعد Fine-Tuning"""
    gif_base64 = generate_play_gif(ft_model, max_steps=400, fps=15)
    return {"gif": gif_base64}

if __name__ == "__main__":
    print("="*70)
    print("🚀 Acrobot Web Interface")
    print("🌐 افتح المتصفح على: http://localhost:8000")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000)