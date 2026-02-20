import React, { useCallback, useEffect, useRef } from 'react';

const getColorForSignalType = (signalType) => {
  switch (signalType?.toLowerCase()) {
    case 'power': return '#FF0000';
    case 'ground': return '#00FF00';
    case 'clock': return '#FFFF00';
    case 'reset': return '#FF00FF';
    case 'analog': return '#00FFFF';
    case 'signal': return '#0000FF';
    default: return '#808080';
  }
};

const LayoutViewer = ({ pcbData, loading }) => {
  const canvasRef = useRef(null);

  const renderLayout = useCallback((ctx, width, height, graph) => {
    if (!graph) return;

    const scale = Math.min(width / 200, height / 200) * 0.8;
    const offsetX = width / 2;
    const offsetY = height / 2;

    ctx.strokeStyle = '#8B4513';
    ctx.lineWidth = 2;

    Object.values(graph.edges || {}).forEach((edge) => {
      const source = graph.nodes[edge.source]?.position;
      const target = graph.nodes[edge.target]?.position;

      if (source && target) {
        ctx.beginPath();
        ctx.moveTo(offsetX + source[0] * scale, offsetY + source[1] * scale);
        ctx.lineTo(offsetX + target[0] * scale, offsetY + target[1] * scale);
        ctx.stroke();
      }
    });

    Object.entries(graph.nodes || {}).forEach(([id, node]) => {
      const [x, y] = node.position;
      ctx.fillStyle = getColorForSignalType(node.signal_type);
      ctx.beginPath();
      ctx.arc(offsetX + x * scale, offsetY + y * scale, 6, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = '#000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(id, offsetX + x * scale, offsetY + y * scale - 10);
    });
  }, []);

  useEffect(() => {
    if (!pcbData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderLayout(ctx, canvas.width, canvas.height, pcbData);
  }, [pcbData, loading, renderLayout]);

  return (
    <div className="border rounded-lg overflow-hidden">
      {loading ? (
        <div className="flex items-center justify-center h-96 bg-gray-50">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Generating PCB layout...</p>
          </div>
        </div>
      ) : pcbData ? (
        <>
          <canvas ref={canvasRef} className="w-full h-96 bg-gray-50" />
          <div className="mt-4 text-sm text-gray-600">
            <div className="flex items-center mb-1"><span className="inline-block w-3 h-3" style={{ backgroundColor: '#FF0000', marginRight: '0.5rem' }}></span> Power</div>
            <div className="flex items-center mb-1"><span className="inline-block w-3 h-3" style={{ backgroundColor: '#00FF00', marginRight: '0.5rem' }}></span> Ground</div>
            <div className="flex items-center mb-1"><span className="inline-block w-3 h-3" style={{ backgroundColor: '#0000FF', marginRight: '0.5rem' }}></span> Signal</div>
            <div className="flex items-center mb-1"><span className="inline-block w-3 h-3" style={{ backgroundColor: '#FFFF00', marginRight: '0.5rem' }}></span> Clock</div>
            <div className="flex items-center"><span className="inline-block w-3 h-3" style={{ backgroundColor: '#00FFFF', marginRight: '0.5rem' }}></span> Analog</div>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-gray-400 text-4xl mb-4">🔌</div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">Generate PCB Layout</h3>
            <p className="text-gray-500">Use the specifications form to generate your first PCB layout</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LayoutViewer;
