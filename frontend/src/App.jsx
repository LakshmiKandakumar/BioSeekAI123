import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Search, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown"; 
import logo from "./assets/bioseeklogo.png";
import './App.css';

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hi 👋 How can I help you with PubMed search?" }
  ]);
  const [input, setInput] = useState("");
  const [IsInitial, setIsInitial] = useState(false);

  const [animateResults, setAnimateResults] = useState(false);
  const resultsRef = useRef(null);

  useEffect(() => {
    resultsRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [results]);

  // ✅ API URL with fallback
  const API_URL = "http://127.0.0.1:8000";


  // ---------------- SEARCH ----------------
  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setAnimateResults(false);
    setIsInitial(true);

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      setResults(data.results || []);
      setAnimateResults(true);
    } catch (error) {
      console.error(error);
      alert("Failed to fetch results.");
    }
    setLoading(false);
  };

  // ---------------- CHAT ----------------
  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = input;
    setMessages(prev => [...prev, { from: "user", text: userMessage }]);
    setInput("");

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();
      setMessages(prev => [
        ...prev,
        { from: "bot", text: data.reply || "No response" }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { from: "bot", text: "❌ Failed to connect to backend." }
      ]);
    }
  };

  // ---------------- FILE UPLOAD ----------------
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setMessages(prev => [...prev, { from: "user", text: `📎 Uploaded: ${file.name}` }]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setMessages(prev => [
        ...prev,
        { from: "bot", text: data.summary || "⚠ Could not summarize the file." }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        { from: "bot", text: "⚠ Error uploading file." }
      ]);
    }
  };

  // ---------------- COMPONENTS ----------------
  const ResultCard = ({ res, animate }) => (
    <motion.div
      initial={animate ? { opacity: 0, y: 20 } : {}}
      animate={animate ? { opacity: 1, y: 0 } : {}}
      transition={animate ? { duration: 0.4 } : {}}
      style={{
        background: "#0d1117",
        margin: "12px 0",
        padding: "18px",
        borderRadius: "12px",
        boxShadow: "0 4px 14px rgba(0,184,169,0.4)",
        border: "1px solid #00B8A9",
        color: "#c9d1d9"
      }}
    >
      {["title", "authors", "year", "abstract", "link"].map((key) => (
        <p key={key} style={{ marginBottom: "8px" }}>
          <b style={{ color: "#00B8A9" }}>
            {key.charAt(0).toUpperCase() + key.slice(1)}:
          </b>{" "}
          {key === "link" ? (
            <a
              href={res[key]}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: "#00B8A9",
                fontWeight: "bold",
                textDecoration: "none"
              }}
            >
              {res[key] || "N/A"}
            </a>
          ) : (
            <span>{res[key] || "N/A"}</span>
          )}
        </p>
      ))}
    </motion.div>
  );

  // ---------------- RENDER ----------------
  return (
    <div style={{
      fontFamily: "'Segoe UI', sans-serif",
      minHeight: "100vh",
      width: "100vw",
      margin: 0,
      padding: 0,
      background: "linear-gradient(135deg, #E0F7F5, #F5F5F5)",
      color: "#2C3E50",
      display: "flex",
      flexDirection: "column",
      position: "relative",
      overflow: "hidden"
    }}>

      {/* Logo */}
      <div style={{ 
        position: "absolute", 
        top: window.innerWidth <= 768 ? "10px" : "20px", 
        left: window.innerWidth <= 768 ? "10px" : "20px", 
        zIndex: 10 
      }}>
        <motion.img
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          src={logo}
          alt="BioSeek Logo"
          style={{ 
            height: window.innerWidth <= 768 ? "120px" : "170px", 
            objectFit: "contain", 
            filter: "drop-shadow(0 0 8px rgba(9, 12, 12, 0.8))" 
          }}
        />
      </div>

      {/* Title */}
      <div style={{ 
        textAlign: "center", 
        marginTop: window.innerWidth <= 768 ? "80px" : "120px", 
        zIndex: 1,
        padding: window.innerWidth <= 768 ? "0 20px" : "0"
      }}>
        <motion.h1
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          style={{
            fontFamily: "'Inter', sans-serif",
            fontSize: window.innerWidth <= 768 ? "2rem" : "3rem",
            margin: 0,
            color: "#02080fff",
            textShadow: "0 0 14px rgba(0,184,169,0.9)"
          }}
        >
          BioSeek AI
        </motion.h1>
        <p style={{ 
          color: "#030008ff", 
          fontSize: window.innerWidth <= 768 ? "0.9rem" : "1rem", 
          marginTop: "4px" 
        }}>
          Accelerating Biomedical Discovery with AI
        </p>
      </div>

      {/* Search */}
      <div style={{
        display: "flex",
        flexDirection: window.innerWidth <= 768 ? "column" : "row",
        justifyContent: "center",
        margin: "20px auto 30px auto",
        width: "100%",
        maxWidth: "700px",
        zIndex: 1,
        padding: window.innerWidth <= 768 ? "0 20px" : "0"
      }}>
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setAnimateResults(false); }}
          placeholder="Discover biomedical insights…"
          style={{
            flex: 1,
            padding: "12px 16px",
            borderRadius: window.innerWidth <= 768 ? "12px 12px 0 0" : "12px 0 0 12px",
            border: "1px solid #00B8A9",
            outline: "none",
            fontSize: "16px",
            background: "#161b22",
            color: "white"
          }}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <motion.button
          onClick={() => !loading && handleSearch()}
          whileHover={{ scale: 1.05, boxShadow: "0 0 20px #00B8A9" }}
          style={{
            padding: "12px 18px",
            borderRadius: window.innerWidth <= 768 ? "0 0 12px 12px" : "0 12px 12px 0",
            border: "none",
            backgroundColor: "#00B8A9",
            color: "white",
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: "16px",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          {loading ? (
            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
              <Loader2 size={24} />
            </motion.div>
          ) : (
            <Search size={24} />
          )}
        </motion.button>
      </div>

      {/* Results */}
      <div style={{ 
        maxWidth: "850px", 
        margin: "0 auto", 
        flex: 1,
        padding: window.innerWidth <= 768 ? "0 20px" : "0"
      }} ref={resultsRef}>
        {results.length > 0 && <p style={{ textAlign: "center", color: "#c9d1d9", marginBottom: "15px" }}>
          <b>{results.length}</b> result{results.length > 1 ? "s" : ""}
        </p>}
        {results.length === 0 && !loading && query && IsInitial && (
          <p style={{ textAlign: "center", color: "#c9d1d9" }}>No results found.</p>
        )}
        {results.map((res, i) => <ResultCard key={i} res={res} animate={animateResults} />)}
      </div>

      {/* Floating Chat Icon */}
      {!chatOpen && (
        <div
          className="chat-icon"
          onClick={() => { setChatOpen(true); setAnimateResults(false); }}
          style={{
            position: "fixed",
            bottom: window.innerWidth <= 768 ? "20px" : "100px",
            right: window.innerWidth <= 768 ? "20px" : "40px",
            zIndex: 999,
            background: "#90d7d1ff",
            color: "#fff",
            borderRadius: "50%",
            width: window.innerWidth <= 768 ? "50px" : "60px",
            height: window.innerWidth <= 768 ? "50px" : "60px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            fontSize: window.innerWidth <= 768 ? "20px" : "28px",
            boxShadow: "0 0 15px #00B8A9, 0 0 30px #00B8A9, 0 0 45px #00B8A9",
            animation: "glow 1.5s infinite alternate"
          }}
        >
          🤖
        </div>
      )}

      {/* Floating Chatbot */}
      {chatOpen && (
        <div className="chat-wrapper" style={{
          position: "fixed",
          bottom: window.innerWidth <= 768 ? "0" : "100px",
          right: window.innerWidth <= 768 ? "0" : "40px",
          width: window.innerWidth <= 768 ? "100vw" : "380px",
          height: window.innerWidth <= 768 ? "100vh" : "auto",
          maxHeight: window.innerWidth <= 768 ? "100vh" : "75vh",
          background: "#161b22",
          borderRadius: window.innerWidth <= 768 ? "0" : "20px",
          boxShadow: "0 0 20px #c8e9e6ff, 0 0 40px #8dd8d2ff, 0 0 60px #90dbd4ff",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          zIndex: 1000,
          padding: window.innerWidth <= 768 ? "10px" : "20px",
          border: "2px solid #00B8A9"
        }}>
          <div className="chat-header" style={{
            padding: window.innerWidth <= 768 ? "8px 12px" : "12px 16px",
            background: "#00B8A9",
            color: "#fff",
            fontWeight: "bold",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderRadius: window.innerWidth <= 768 ? "0" : "12px 12px 0 0",
            boxShadow: "0 0 10px #00B8A9"
          }}>
            Clario AI
            <span className="close-btn" onClick={() => { setChatOpen(false); setAnimateResults(false); }} style={{ cursor: "pointer" }}>
              ✖
            </span>
          </div>

          <div className="chat-body" style={{ 
            flex: 1, 
            padding: window.innerWidth <= 768 ? "10px" : "15px", 
            overflowY: "auto" 
          }}>
            <div className="messages" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {messages.map((msg, i) => (
                <div key={i} style={{ display: "flex", justifyContent: msg.from === "user" ? "flex-end" : "flex-start" }}>
                  <div className={msg.from === "user" ? "user-msg" : "bot-msg"} style={{
                    display: "inline-block",
                    padding: window.innerWidth <= 768 ? "8px 12px" : "10px 14px",
                    borderRadius: "14px",
                    background: msg.from === "user" ? "#00B8A9" : "#0d1117",
                    color: "#fff",
                    maxWidth: window.innerWidth <= 768 ? "85%" : "75%",
                    whiteSpace: "pre-wrap",
                    wordWrap: "break-word",
                    boxShadow: msg.from === "user" ? "0 0 10px #00B8A9" : "0 0 6px #00B8A9",
                    fontSize: window.innerWidth <= 768 ? "14px" : "16px"
                  }}>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="input-container" style={{ 
            display: "flex", 
            padding: window.innerWidth <= 768 ? "8px" : "10px", 
            borderTop: "1px solid #00B8A9", 
            gap: "8px" 
          }}>
            <label className="upload-btn" style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
              📎
              <input type="file" style={{ display: "none" }} onChange={handleFileUpload} />
            </label>
            <input
              type="text"
              placeholder="Type a message..."
              value={input}
              onChange={(e) => { setInput(e.target.value); setAnimateResults(false); }}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              style={{
                flex: 1,
                padding: window.innerWidth <= 768 ? "10px" : "14px",
                borderRadius: "14px",
                border: "none",
                background: "#0d1117",
                color: "#fff",
                fontSize: window.innerWidth <= 768 ? "14px" : "14px",
                boxShadow: "0 0 8px #00B8A9 inset"
              }}
            />
            
            <button onClick={sendMessage} style={{
              padding: window.innerWidth <= 768 ? "10px 14px" : "12px 18px",
              background: "#00B8A9",
              border: "none",
              borderRadius: "14px",
              color: "#fff",
              cursor: "pointer",
              fontWeight: "bold",
              boxShadow: "0 0 12px #91cfcaff",
              fontSize: window.innerWidth <= 768 ? "14px" : "16px"
            }}>Send</button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        textAlign: "center",
        padding: window.innerWidth <= 768 ? "15px" : "20px",
        marginTop: "50px",
        background: "rgba(0,0,0,0.8)",
        color: "#fff",
        fontSize: window.innerWidth <= 768 ? "12px" : "14px",
        borderTop: "1px solid rgba(0, 184, 169, 0.4)"
      }}>
        <p style={{ 
          maxWidth: "950px", 
          margin: "0 auto", 
          lineHeight: "1.6",
          padding: window.innerWidth <= 768 ? "0 20px" : "0"
        }}>
          <a href="https://www.ncbi.nlm.nih.gov/" target="_blank" rel="noopener noreferrer" style={{ color: "#00B8A9", fontWeight: "bold", textDecoration: "none" }}>
            NCBI
          </a>{" "}
          (National Center for Biotechnology Information) is part of the U.S. National Library of Medicine.
        </p>
      </footer>
    </div>
  );
}

