using System;
using IO.Milvus.Client;
using Microsoft.SemanticKernel.Memory;

namespace Microsoft.SemanticKernel.Connectors.Memory.Milvus;

/// <summary>
/// Milvus memory store interface
/// </summary>
public interface IMilvusMemoryStore : IMemoryStore, IDisposable
{
    /// <summary>
    /// Milvus client that allows you to visit most of milvus api.
    /// </summary>
    /// <remarks>
    /// <see href="https://github.com/milvus-io/milvus-sdk-csharp"/>
    /// </remarks>
    IMilvusClient MilvusClient { get; }
}
