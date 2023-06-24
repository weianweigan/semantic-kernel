// Copyright (c) Microsoft. All rights reserved.

using IO.Milvus;
using Microsoft.SemanticKernel.Memory;

namespace Microsoft.SemanticKernel.Connectors.Memory.Milvus;

/// <summary>
/// Milvus connector config that allow user control more details about milvus.
/// And user can choose a config quickly.
/// </summary>
public sealed class MilvusConnectorConfig
{
    /// <summary>
    /// Whether to enable Automatic ID (primary key) allocation or not
    /// </summary>
    /// <remarks>
    /// <para>
    /// <para>
    /// For <see cref="MemoryRecordMetadata.Id"/>
    /// </para>
    /// <para>
    /// <seealso href="https://milvus.io/docs/schema.md"/>
    /// </para>
    /// </para>
    /// </remarks>
    public bool AutoId { get; set; } = false;

    /// <summary>
    /// <see href="https://milvus.io/docs/index.md"/>
    /// </summary>
    public MilvusIndexType IndexType { get; set; } = MilvusIndexType.IVF_FLAT;

    /// <summary>
    /// Use json field to store <see cref="MemoryRecordMetadata"/>
    /// </summary>
    /// <remarks>
    /// <see href="https://milvus.io/docs/json_data_type.md"/>
    /// </remarks>
    public bool JsonFieldForMetadata { get; set; } = false;

    /// <summary>
    /// Config for zilliz cloud.
    /// </summary>
    /// <remarks>
    /// <see href="https://docs.zilliz.com/"/>
    /// </remarks>
    /// <returns>Instance of <see cref="MilvusConnectorConfig"/></returns>
    public static MilvusConnectorConfig ForZillizCloud()
    {
        return new MilvusConnectorConfig
        {
            AutoId = true,
            IndexType = MilvusIndexType.AUTOINDEX,
            JsonFieldForMetadata = true
        };
    }

    /// <summary>
    /// Config for milvus version v2.2.9 and above.
    /// </summary>
    /// <returns>Instance of <see cref="MilvusConnectorConfig"/></returns>
#pragma warning disable CA1707 // Identifiers should not contain underscores.
    public static MilvusConnectorConfig GreaterOrEqualV2_2_9()
#pragma warning restore CA1707 // Identifiers should not contain underscores.
    {
        return new MilvusConnectorConfig()
        {
            AutoId = true,
            IndexType = MilvusIndexType.AUTOINDEX,
            JsonFieldForMetadata = true
        };
    }
}
