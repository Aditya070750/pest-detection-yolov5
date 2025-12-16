// Optimized live update script
document.addEventListener("DOMContentLoaded", () => {
  if (window.location.pathname === "/live") {
    console.log("🔁 Live feed auto-update enabled");

    // Function to refresh detection data
    async function refreshLiveData() {
      try {
        const res = await fetch("/get_log_data");
        if (!res.ok) throw new Error("Failed to fetch data");
        const data = await res.json();

        if (data.length > 0) {
          const latest = data[data.length - 1];
          document.getElementById("object-name").textContent = latest.object || "None";
          document.getElementById("last-update").textContent = "Last Update: " + (latest.time || "N/A");
        }
      } catch (err) {
        console.error("⚠️ Live data update failed:", err);
      }
    }

    // Run immediately and every 10s
    refreshLiveData();
    setInterval(refreshLiveData, 10000);
  }
});
