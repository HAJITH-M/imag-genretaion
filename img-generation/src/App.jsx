import { useState } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState(""); // State to hold the user input
  const [imageData, setImageData] = useState(null); // State to hold the generated image
  const [error, setError] = useState(""); // State to hold any errors

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(""); // Reset error on new submission
    setImageData(null); // Reset image on new submission

    // Make the POST request to the Flask API
    try {
      const response = await fetch("http://192.168.127.88:5000/generate-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();

      if (data.image) {
        setImageData(data.image); // Set the image data to state
      } else {
        setError("Failed to generate image.");
      }
    } catch (err) {
      setError("Error connecting to the backend.");
    }
  };

  return (
    <div className="App">
      <h1>Image Generator</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a description to generate an image"
          rows="4"
          cols="50"
        />
        <br />
        <button type="submit">Generate Image</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {imageData && (
  <div>
    <h2>Generated Image:</h2>
    <img src={`data:image/png;base64,${imageData}`} alt="Generated" />
  </div>
)}

    </div>
  );
}

export default App;
