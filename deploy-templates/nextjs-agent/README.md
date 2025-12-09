# AI Sports News Channel - Streaming Avatar

Real-time AI sports anchor using HeyGen's Streaming Avatar SDK.

**Modern AI Pro Workshop** | Build Your Own ESPN

## Features

- Real-time streaming AI avatar
- Multiple anchor personas (Sports Anchor, News Reporter, Friendly Host)
- Custom script input
- Live broadcasting indicator
- Debug logging

## Quick Start

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env.local
   ```
   Add your HeyGen API key to `.env.local`

3. **Run development server**
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000)

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/streaming-avatar)

Add `HEYGEN_API_KEY` as an environment variable in Vercel.

## How It Works

1. Click "Start Session" to initialize the avatar
2. Enter your script in the text area
3. Click "Speak Script" to have the avatar read it
4. Use "Stop" to interrupt, "End Session" to close

## API Keys

Get your HeyGen API key at [heygen.com](https://heygen.com) (Settings > API)

Free tier includes trial credits for testing.

## Tech Stack

- Next.js 16
- TypeScript
- Tailwind CSS
- @heygen/streaming-avatar SDK
- WebRTC

## Workshop

This is part of the **Modern AI Pro Agentic AI Workshop**.

What you're building that ChatGPT/Claude can't do:
- Real-time interactive AI avatars
- Custom streaming video generation
- Low-latency WebRTC communication
