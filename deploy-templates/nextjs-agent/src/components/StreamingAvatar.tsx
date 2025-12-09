"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import StreamingAvatar, {
  AvatarQuality,
  StreamingEvents,
  TaskType,
  VoiceEmotion,
} from "@heygen/streaming-avatar";

// Chroma key function to remove green screen
function applyChromaKey(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  threshold: number = 100
) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const red = data[i];
    const green = data[i + 1];
    const blue = data[i + 2];

    // Detect green screen pixels
    if (green > threshold && red < threshold && blue < threshold) {
      data[i + 3] = 0; // Set alpha to transparent
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

interface Match {
  id: string;
  name: string;
  sport: string;
  time: string;
}

// Separate avatar options (visual appearance)
const AVATARS: Record<string, { id: string; description: string }> = {
  "Wayne (Male, Professional)": { id: "Wayne_20240711", description: "Professional male anchor" },
  "Anna (Female, News)": { id: "Anna_public_3_20240108", description: "Female news reporter" },
  "Josh (Male, Casual)": { id: "josh_lite3_20230714", description: "Friendly male host" },
  "Tyler (Male, Suit)": { id: "Tyler-incasualsuit-20220721", description: "Professional in suit" },
  "Kayla (Female, Casual)": { id: "Kayla-incasualsuit-20220818", description: "Casual female host" },
};

// Separate voice options (audio)
const VOICES: Record<string, { id: string; emotion: VoiceEmotion; description: string }> = {
  "Male - Excited": { id: "1bd001e7e50f421d891986aad5158bc8", emotion: VoiceEmotion.EXCITED, description: "Energetic male voice" },
  "Female - Broadcaster": { id: "2d5b0e6cf36f460aa7fc47e3eee4ba54", emotion: VoiceEmotion.BROADCASTER, description: "Professional female voice" },
  "Male - Friendly": { id: "131a436c47064f708210df6628ef8f32", emotion: VoiceEmotion.FRIENDLY, description: "Warm male voice" },
  "Male - Serious": { id: "1bd001e7e50f421d891986aad5158bc8", emotion: VoiceEmotion.SERIOUS, description: "Authoritative male voice" },
  "Female - Soothing": { id: "2d5b0e6cf36f460aa7fc47e3eee4ba54", emotion: VoiceEmotion.SOOTHING, description: "Calm female voice" },
};

// Sample matches - in production, fetch from Odds API
const SAMPLE_MATCHES: Match[] = [
  { id: "1", name: "Lakers vs Celtics", sport: "NBA", time: "Tonight 7:30 PM" },
  { id: "2", name: "Chiefs vs Bills", sport: "NFL", time: "Sunday 4:25 PM" },
  { id: "3", name: "Man City vs Liverpool", sport: "Premier League", time: "Saturday 12:30 PM" },
  { id: "4", name: "Yankees vs Red Sox", sport: "MLB", time: "Tomorrow 7:05 PM" },
  { id: "5", name: "India vs Australia", sport: "Cricket", time: "Friday 9:30 AM" },
];

export default function StreamingAvatarComponent() {
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [selectedMatch, setSelectedMatch] = useState<Match>(SAMPLE_MATCHES[0]);
  const [scriptText, setScriptText] = useState("");
  const [selectedAvatar, setSelectedAvatar] = useState("Wayne (Male, Professional)");
  const [selectedVoice, setSelectedVoice] = useState("Male - Excited");
  const [error, setError] = useState<string | null>(null);
  const [debug, setDebug] = useState<string[]>([]);
  const [removeGreenScreen, setRemoveGreenScreen] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const avatarRef = useRef<StreamingAvatar | null>(null);
  const animationRef = useRef<number | null>(null);

  // Generate script based on selected match
  useEffect(() => {
    const match = selectedMatch;
    setScriptText(
      `Welcome to AI Sports News! I'm bringing you the latest on tonight's big matchup. ` +
      `The ${match.name} game is set for ${match.time}. ` +
      `This ${match.sport} showdown promises to be an exciting one. ` +
      `Both teams have been performing well this season, and fans are expecting a close contest. ` +
      `Stay tuned for live updates and analysis right here on your AI Sports Channel.`
    );
  }, [selectedMatch]);

  const addDebug = useCallback((msg: string) => {
    setDebug((prev) => [...prev.slice(-9), `${new Date().toLocaleTimeString()}: ${msg}`]);
  }, []);

  // Chroma key rendering loop
  const renderChromaKey = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas size to video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
    }

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Apply chroma key if enabled
    if (removeGreenScreen) {
      applyChromaKey(ctx, canvas.width, canvas.height, 90);
    }

    animationRef.current = requestAnimationFrame(renderChromaKey);
  }, [removeGreenScreen]);

  // Start/stop chroma key rendering
  useEffect(() => {
    if (isSessionActive && removeGreenScreen) {
      animationRef.current = requestAnimationFrame(renderChromaKey);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isSessionActive, removeGreenScreen, renderChromaKey]);

  const fetchAccessToken = async (): Promise<string> => {
    const response = await fetch("/api/heygen", { method: "POST" });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to get access token");
    }
    const data = await response.json();
    return data.token;
  };

  const startSession = async () => {
    setIsLoading(true);
    setError(null);
    addDebug("Starting session...");

    try {
      const token = await fetchAccessToken();
      addDebug("Got access token");

      const avatar = new StreamingAvatar({ token });
      avatarRef.current = avatar;

      const avatarConfig = AVATARS[selectedAvatar];
      const voiceConfig = VOICES[selectedVoice];

      // Set up event listeners
      avatar.on(StreamingEvents.STREAM_READY, async (event) => {
        addDebug("Stream ready!");
        if (videoRef.current && event.detail) {
          videoRef.current.srcObject = event.detail;
          videoRef.current.play().catch(console.error);
        }
        setIsSessionActive(true);
        setIsLoading(false);

        // Auto-speak the script after a short delay
        setTimeout(async () => {
          if (avatarRef.current && scriptText.trim()) {
            addDebug("Auto-speaking script...");
            try {
              await avatarRef.current.speak({
                text: scriptText,
                taskType: TaskType.REPEAT,
              });
              addDebug("Auto-speak started");
            } catch (err) {
              addDebug("Auto-speak failed");
            }
          }
        }, 1000);
      });

      avatar.on(StreamingEvents.STREAM_DISCONNECTED, () => {
        addDebug("Stream disconnected");
        setIsSessionActive(false);
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
      });

      avatar.on(StreamingEvents.AVATAR_START_TALKING, () => {
        addDebug("Avatar started talking");
        setIsSpeaking(true);
      });

      avatar.on(StreamingEvents.AVATAR_STOP_TALKING, () => {
        addDebug("Avatar stopped talking");
        setIsSpeaking(false);
      });

      addDebug(`Creating session with avatar: ${avatarConfig.id}, voice: ${selectedVoice}`);

      await avatar.createStartAvatar({
        quality: AvatarQuality.High,
        avatarName: avatarConfig.id,
        voice: {
          voiceId: voiceConfig.id,
          rate: 1.0,
          emotion: voiceConfig.emotion,
        },
        language: "en",
      });

      addDebug("Session created successfully");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      addDebug(`Error: ${message}`);
      setIsLoading(false);
    }
  };

  const speak = async () => {
    if (!avatarRef.current || !scriptText.trim()) return;

    addDebug("Sending speech request...");
    try {
      await avatarRef.current.speak({
        text: scriptText,
        taskType: TaskType.REPEAT,
      });
      addDebug("Speech request sent");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Speech failed";
      setError(message);
      addDebug(`Speech error: ${message}`);
    }
  };

  const stopSpeaking = async () => {
    if (!avatarRef.current) return;
    try {
      await avatarRef.current.interrupt();
      addDebug("Interrupted speech");
    } catch (err) {
      addDebug("Interrupt failed");
    }
  };

  const endSession = async () => {
    if (!avatarRef.current) return;

    addDebug("Ending session...");
    try {
      await avatarRef.current.stopAvatar();
    } catch (err) {
      addDebug("Stop avatar error (may be normal)");
    }

    avatarRef.current = null;
    setIsSessionActive(false);
    setIsSpeaking(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    addDebug("Session ended");
  };

  useEffect(() => {
    return () => {
      if (avatarRef.current) {
        avatarRef.current.stopAvatar().catch(console.error);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            AI Sports News Channel
          </h1>
          <p className="text-slate-400">
            Powered by HeyGen Streaming Avatar | Modern AI Pro Workshop
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Video Panel */}
          <div className="bg-slate-800 rounded-2xl p-6 shadow-2xl">
            {/* Video container with studio background */}
            <div className="aspect-video bg-gradient-to-b from-blue-900 via-slate-800 to-slate-900 rounded-xl overflow-hidden relative">
              {/* News studio background */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-900/80 via-slate-800 to-slate-900" />
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl" />
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-cyan-500/10 rounded-full blur-2xl" />

              {/* Video element for stream source (hidden when using chroma key, but NOT muted for audio) */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className={removeGreenScreen ? "absolute w-1 h-1 opacity-0" : "w-full h-full object-contain relative z-10"}
              />

              {/* Canvas for chroma key output */}
              <canvas
                ref={canvasRef}
                className={removeGreenScreen ? "w-full h-full object-contain relative z-10" : "hidden"}
              />
              {!isSessionActive && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-slate-400">
                    <div className="text-6xl mb-4">🎬</div>
                    <p>Start a session to see your AI anchor</p>
                  </div>
                </div>
              )}
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <div className="text-center text-white">
                    <div className="animate-spin text-4xl mb-4">⏳</div>
                    <p>Initializing avatar...</p>
                  </div>
                </div>
              )}
              {isSpeaking && (
                <div className="absolute top-4 right-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm animate-pulse">
                  LIVE
                </div>
              )}
              {/* Match ticker */}
              {isSessionActive && (
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-r from-blue-900 to-blue-800 text-white px-4 py-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold">{selectedMatch.sport}</span>
                    <span>{selectedMatch.name}</span>
                    <span className="text-blue-300">{selectedMatch.time}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="mt-6 space-y-4">
              {/* Match Selection */}
              <div>
                <label className="block text-sm text-slate-400 mb-2">
                  Select Match
                </label>
                <select
                  value={selectedMatch.id}
                  onChange={(e) => {
                    const match = SAMPLE_MATCHES.find(m => m.id === e.target.value);
                    if (match) setSelectedMatch(match);
                  }}
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2"
                >
                  {SAMPLE_MATCHES.map((match) => (
                    <option key={match.id} value={match.id}>
                      {match.sport}: {match.name} - {match.time}
                    </option>
                  ))}
                </select>
              </div>

              {/* Avatar Selection (Visual) */}
              <div>
                <label className="block text-sm text-slate-400 mb-2">
                  Select Avatar (Appearance)
                </label>
                <select
                  value={selectedAvatar}
                  onChange={(e) => setSelectedAvatar(e.target.value)}
                  disabled={isSessionActive}
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 disabled:opacity-50"
                >
                  {Object.entries(AVATARS).map(([name, config]) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500 mt-1">{AVATARS[selectedAvatar]?.description}</p>
              </div>

              {/* Voice Selection (Audio) */}
              <div>
                <label className="block text-sm text-slate-400 mb-2">
                  Select Voice (Audio)
                </label>
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  disabled={isSessionActive}
                  className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 disabled:opacity-50"
                >
                  {Object.entries(VOICES).map(([name, config]) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-slate-500 mt-1">{VOICES[selectedVoice]?.description}</p>
              </div>

              {/* Green screen toggle */}
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id="removeGreenScreen"
                  checked={removeGreenScreen}
                  onChange={(e) => setRemoveGreenScreen(e.target.checked)}
                  className="w-4 h-4 rounded"
                />
                <label htmlFor="removeGreenScreen" className="text-sm text-slate-400">
                  Remove green screen background
                </label>
              </div>

              <div className="flex gap-4">
                {!isSessionActive ? (
                  <button
                    onClick={startSession}
                    disabled={isLoading}
                    className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    {isLoading ? "Starting..." : "Start Session"}
                  </button>
                ) : (
                  <button
                    onClick={endSession}
                    className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    End Session
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Script Panel */}
          <div className="bg-slate-800 rounded-2xl p-6 shadow-2xl">
            <h2 className="text-xl font-semibold text-white mb-4">
              Anchor Script
            </h2>

            <textarea
              value={scriptText}
              onChange={(e) => setScriptText(e.target.value)}
              placeholder="Enter the script for your AI anchor to read..."
              className="w-full h-48 bg-slate-700 text-white rounded-lg px-4 py-3 resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />

            <div className="mt-2 text-sm text-slate-400">
              {scriptText.split(" ").filter(Boolean).length} words |{" "}
              ~{Math.ceil(scriptText.split(" ").filter(Boolean).length / 150 * 60)} seconds
            </div>

            <div className="mt-4 flex gap-4">
              <button
                onClick={speak}
                disabled={!isSessionActive || isSpeaking || !scriptText.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                {isSpeaking ? "Speaking..." : "Speak Script"}
              </button>
              <button
                onClick={stopSpeaking}
                disabled={!isSpeaking}
                className="bg-orange-600 hover:bg-orange-700 disabled:bg-slate-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                Stop
              </button>
            </div>

            {error && (
              <div className="mt-4 bg-red-900/50 border border-red-500 text-red-200 rounded-lg p-4">
                {error}
              </div>
            )}

            {/* Debug Log */}
            <div className="mt-6">
              <h3 className="text-sm font-semibold text-slate-400 mb-2">
                Debug Log
              </h3>
              <div className="bg-slate-900 rounded-lg p-3 h-32 overflow-y-auto text-xs text-slate-400 font-mono">
                {debug.length === 0 ? (
                  <p>Waiting for events...</p>
                ) : (
                  debug.map((msg, i) => <div key={i}>{msg}</div>)
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-slate-500 text-sm">
          <p>
            Built for Modern AI Pro Workshop | Using HeyGen Streaming Avatar SDK
          </p>
        </div>
      </div>
    </div>
  );
}
