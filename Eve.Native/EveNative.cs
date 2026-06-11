using System;
using System.Runtime.InteropServices;

namespace Eve.Native
{
    public class EveNativeHandle : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed = false;

        public EveNativeHandle(int vocabSize, int embedDim, int numHeads, int numLayers,
                               int feedForwardDim, int maxSeqLen)
        {
            _handle = eve_create(vocabSize, embedDim, numHeads, numLayers, feedForwardDim, maxSeqLen);
            if (_handle == IntPtr.Zero)
                throw new Exception("Failed to create native handle");
        }

        public void InitWeights(uint seed)
        {
            eve_init_weights(_handle, seed);
        }

        public void SetLearningRate(float lr)
        {
            eve_set_learning_rate(_handle, lr);
        }

        public float TrainStep(int[] tokens, int seqLen)
        {
            return eve_train_step(_handle, tokens, seqLen);
        }

        public int Generate(int[] promptTokens, int promptLen, int[] outputTokens,
                           int maxOutputLen, float temperature)
        {
            return eve_generate(_handle, promptTokens, promptLen, outputTokens, maxOutputLen, temperature);
        }

        public int SaveModel(string path)
        {
            return eve_save_model(_handle, path);
        }

        public int LoadModel(string path)
        {
            return eve_load_model(_handle, path);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    eve_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~EveNativeHandle()
        {
            Dispose(false);
        }

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr eve_create(int vocabSize, int embedDim, int numHeads,
                                                int numLayers, int feedForwardDim, int maxSeqLen);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern void eve_destroy(IntPtr handle);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern void eve_init_weights(IntPtr handle, uint seed);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern void eve_set_learning_rate(IntPtr handle, float lr);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern float eve_train_step(IntPtr handle, int[] tokens, int seqLen);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern int eve_generate(IntPtr handle, int[] promptTokens, int promptLen,
                                               int[] outputTokens, int maxOutputLen, float temperature);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern int eve_save_model(IntPtr handle, string path);

        [DllImport("eve_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern int eve_load_model(IntPtr handle, string path);
    }
}
