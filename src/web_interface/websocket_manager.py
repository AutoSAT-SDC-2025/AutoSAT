"""
WebSocket manager for real-time CAN data broadcasting.

Provides thread-safe WebSocket client management and CAN message broadcasting
to connected web interface clients with proper error handling and cleanup.
"""

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)

thread_local_data = threading.local()

class CanWebSocketManager:
    """
    Manages WebSocket connections for CAN data broadcasting.
    
    Maintains list of connected clients and handles broadcasting
    of CAN messages with automatic client cleanup on errors.
    """
    
    def __init__(self):
        """Initialize WebSocket manager with empty client list."""
        self.clients = []
    
    async def broadcast(self, json_data):
        """
        Broadcast JSON data to all connected clients.
        
        Args:
            json_data: JSON string to broadcast to all clients
        """
        if not self.clients:
            return
        
        for client in list(self.clients):
            try:
                await client.send_text(json_data)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                if client in self.clients:
                    self.clients.remove(client)
    
    def add_client(self, websocket):
        """
        Add WebSocket client to receive CAN messages.
        
        Args:
            websocket: WebSocket connection to add
        """
        self.clients.append(websocket)
        logger.info(f"CAN WebSocket client added, total clients: {len(self.clients)}")
    
    def remove_client(self, websocket):
        """
        Remove WebSocket client from broadcast list.
        
        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f"CAN WebSocket client removed, total clients: {len(self.clients)}")

# Global WebSocket manager instance
can_ws_manager = CanWebSocketManager()

async def broadcast_can_json(json_data):
    """
    Broadcast CAN JSON data to all WebSocket clients.
    
    Args:
        json_data: JSON string containing CAN message data
    """
    await can_ws_manager.broadcast(json_data)

def get_thread_event_loop():
    """
    Get or create event loop for current thread.
    
    Creates thread-local event loop for synchronous CAN broadcasting
    from non-async contexts like CAN listener threads.
    
    Returns:
        Event loop for the current thread
    """
    if not hasattr(thread_local_data, "loop"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        thread_local_data.loop = loop
        logger.debug(f"Created new event loop for thread {threading.current_thread().name}")
    return thread_local_data.loop

def sync_broadcast_can_json(json_data):
    """
    Synchronous wrapper to broadcast CAN JSON data.
    
    Allows broadcasting from synchronous contexts like CAN message
    listeners by managing event loop internally.
    
    Args:
        json_data: JSON string containing CAN message data
    """
    if not can_ws_manager.clients:
        logger.debug("No WebSocket clients connected, skipping broadcast")
        return
    
    try:
        loop = get_thread_event_loop()
        loop.run_until_complete(broadcast_can_json(json_data))
    except Exception as e:
        logger.error(f"Error broadcasting CAN data: {e}")