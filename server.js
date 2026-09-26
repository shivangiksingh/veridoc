require("dotenv").config();

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");

const casesRouter = require("./routes/cases");
const auditRouter = require("./routes/audit");
const verificationRouter = require("./routes/verification");

const app = express();
const port = Number(process.env.PORT || 5000);

app.use(helmet({ crossOriginResourcePolicy: false }));
app.use(cors({
  origin: process.env.CLIENT_ORIGIN || "http://localhost:5173",
  credentials: true,
}));
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true, limit: "2mb" }));
app.use(morgan("dev"));

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    service: "veridoc-backend",
    status: "online",
    time: new Date().toISOString(),
  });
});

app.use("/api/cases", casesRouter);
app.use("/api/audit", auditRouter);
app.use("/api/verification", verificationRouter);

app.use((err, req, res, next) => {
  console.error(err);
  const status = err.code === "LIMIT_FILE_SIZE" ? 413 : 500;
  res.status(status).json({
    success: false,
    message: err.message || "Internal server error",
  });
});

app.use((req, res) => {
  res.status(404).json({ success: false, message: "Route not found" });
});

app.listen(port, () => {
  console.log(`Veridoc backend running on http://localhost:${port}`);
});
