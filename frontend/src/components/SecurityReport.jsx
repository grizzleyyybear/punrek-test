import React from 'react';
import './SecurityReport.css';

function SecurityReport({ report }) {
  if (!report) return null;

  const { vulnerabilities, overall_score } = report;

  const status = overall_score >= 0.8 ? 'Passed' : 'Failed';

  return (
    <div className={`security-report ${status.toLowerCase()}`}>
      <h3>3. Security Analysis</h3>
      <div className="summary">
        <strong>Status: {status}</strong>
        <span>({vulnerabilities.length} vulnerabilities found)</span>
        <div className="score">Score: {(overall_score * 100).toFixed(1)}/100</div>
      </div>
      {vulnerabilities.length > 0 && (
        <div className="vulnerability-list">
          <h4>Vulnerabilities:</h4>
          <ul>
            {vulnerabilities.map((vuln, index) => (
              <li key={index}>
                <strong>{vuln.type}:</strong> {vuln.description}
                {vuln.location && <div className="details">Affected Nodes: {JSON.stringify(vuln.location)}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default SecurityReport;
