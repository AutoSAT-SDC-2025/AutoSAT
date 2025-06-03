import asyncio
import logging
import threading

logger = logging.getLogger(__name__)

thread_local_data = threading.local()

class CanWebSocketManager:
    def __init__(self):
        self.clients = []
    
    async def broadcast(self, json_data):
        """Broadcast JSON data to all clients"""
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
        """Add a client to receive CAN messages"""
        self.clients.append(websocket)
        logger.info(f"CAN WebSocket client added, total clients: {len(self.clients)}")
    
    def remove_client(self, websocket):
        """Remove a client"""
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f"CAN WebSocket client removed, total clients: {len(self.clients)}")

can_ws_manager = CanWebSocketManager()

async def broadcast_can_json(json_data):
    """Broadcast CAN JSON data to all WebSocket clients"""
    await can_ws_manager.broadcast(json_data)

def get_thread_event_loop():
    if not hasattr(thread_local_data, "loop"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        thread_local_data.loop = loop
        logger.debug(f"Created new event loop for thread {threading.current_thread().name}")
    return thread_local_data.loop

def sync_broadcast_can_json(json_data):
    """Synchronous wrapper to broadcast CAN JSON data"""
    if not can_ws_manager.clients:
        logger.debug("No WebSocket clients connected, skipping broadcast")
        return
    
    try:
        loop = get_thread_event_loop()

        loop.run_until_complete(broadcast_can_json(json_data))
    except Exception as e:
        logger.error(f"Error broadcasting CAN data: {e}")