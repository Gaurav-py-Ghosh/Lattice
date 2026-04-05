import { useEffect, useRef } from "react";
import {
  Room,
  RoomEvent,
  RemoteTrack,
  Track,
} from "livekit-client";

const CONVFLOW_BACKEND = "http://localhost:8001";
const LIVEKIT_URL = "ws://localhost:7880";

interface UseConvFlowRoomOptions {
  onTurnEnd: () => void;
  stream: MediaStream | null;
}

export function useConvFlowRoom({ onTurnEnd, stream }: UseConvFlowRoomOptions) {
  const onTurnEndRef = useRef(onTurnEnd);

  useEffect(() => {
    onTurnEndRef.current = onTurnEnd;
  }, [onTurnEnd]);

  useEffect(() => {
    // We only connect if the parent has provided a stream
    if (!stream) return;
    const mediaStream = stream;

    const room = new Room();

    room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
      if (track.kind === Track.Kind.Audio) {
        track.attach();
        console.log("🔊 Agent voice playing");
      }
    });

    room.on(RoomEvent.DataReceived, (payload: Uint8Array) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload));
        if (msg.event === "turn_end") {
          onTurnEndRef.current();
        }
      } catch (e) { /* ignore */ }
    });

    async function connect() {
      try {
        const res = await fetch(`${CONVFLOW_BACKEND}/token`);
        const { token } = await res.json();
        await room.connect(LIVEKIT_URL, token);
        console.log("✅ Connected to Agent Room");

        if (mediaStream.getAudioTracks().length > 0) {
          const { LocalAudioTrack } = await import("livekit-client");
          const audioTrack = mediaStream.getAudioTracks()[0];
          await room.localParticipant.publishTrack(new LocalAudioTrack(audioTrack));
          console.log("🎤 Mic is LIVE");
        }
      } catch (err) {
        console.error("❌ Connection failed:", err);
      }
    }

    connect();

    return () => {
      console.log("🧹 Cleanup");
      room.disconnect();
    };
  }, [stream]); // Re-connect simple logic
}
