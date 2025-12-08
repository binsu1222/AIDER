import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080/api';

const stockApi = {
  // 60일 주가 데이터 조회
  getStockPrices: async (stockName, endDate) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stocks/prices`, {
        params: {
          stockName,
          endDate: endDate || undefined
        }
      });
      return response.data;
    } catch (error) {
      console.error('주가 데이터 조회 실패:', error);
      throw error;
    }
  },

  // 종목 데이터 초기화
  initializeStock: async (stockName, stockCode, market = 'KOSPI') => {
    try {
      const response = await axios.post(`${API_BASE_URL}/stocks/initialize`, null, {
        params: {
          stockName,
          stockCode,
          market
        }
      });
      return response.data;
    } catch (error) {
      console.error('데이터 초기화 실패:', error);
      throw error;
    }
  },

  // 저장된 데이터 개수 확인
  getDataCount: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stocks/count`);
      return response.data;
    } catch (error) {
      console.error('데이터 개수 조회 실패:', error);
      throw error;
    }
  }
};

export default stockApi;