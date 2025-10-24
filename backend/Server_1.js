import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import dotenv from "dotenv";

dotenv.config();
const app = express();

app.use(cors());
app.use(bodyParser.json());

const PORT = process.env.PORT || 3000;
const SHEET_WEBHOOK = process.env.SHEET_WEBHOOK;

// ✅ Ruta principal
app.get("/", (req, res) => {
  res.send("✅ Backend de Wallet Multichain activo");
});

// ✅ Ruta para recibir datos del frontend
app.post("/wallet-data", async (req, res) => {
  const data = req.body;
  console.log("📥 Datos recibidos:", data);

  try {
    const response = await fetch(SHEET_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    console.log("📤 Enviado a Google Sheets");
    res.json({ success: true, message: "Datos enviados correctamente" });
  } catch (error) {
    console.error("❌ Error al enviar a Sheets:", error);
    res.status(500).json({ success: false, error: "Error al enviar a Sheets" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Servidor backend corriendo en http://localhost:${PORT}`);
});
